"""MobileFaceNet TFLite Model — Face Embedding Engine for CogniDrive.

This module wraps the MobileFaceNet TensorFlow Lite model for high-performance,
CPU-only facial embedding generation. It is the core biometric inference engine
for CogniDrive's driver authentication and Digital Twin initialization pipeline.

Architecture:
    MobileFaceNet is a compact, depthwise separable CNN architecture trained on
    large face recognition datasets (e.g., MS-Celeb-1M). At inference, it accepts
    a preprocessed RGB face crop and returns a 128-D (or 512-D depending on the
    checkpoint) L2-normalized embedding vector.

    The embedding lies on a hypersphere, so cosine similarity is equivalent to
    the dot product of two unit-norm vectors — making verification both fast and
    geometrically meaningful.

Pipeline::

    RGB face crop (H×W×3, uint8 or float32)
        ↓  preprocess_face()
    Resized, normalized float32 tensor (1×112×112×3)
        ↓  TFLite Interpreter.invoke()
    Raw embedding (1×D)
        ↓  normalize_embedding()
    L2-normalized unit-vector (D,) float32

Usage::

    model = MobileFaceNetModel.get_instance()
    embedding = model.extract_embedding(face_crop_bgr_or_rgb)
    result = model.verify_faces(embedding_a, embedding_b)
    # → {"verified": True, "similarity": 0.87, "threshold": 0.65}

Note:
    All public methods are thread-safe. The TFLite interpreter is protected by
    a ``threading.Lock`` to prevent concurrent invocation from camera and API
    threads. Preprocessing is stateless and lock-free.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("CogniDrive.MobileFaceNet")

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import tflite_runtime.interpreter as tflite  # type: ignore[import]
    _TFLITE_BACKEND = "tflite_runtime"
    _TFLITE_AVAILABLE = True
    logger.info("MobileFaceNet: Using tflite_runtime interpreter backend.")
except ImportError:
    try:
        import tensorflow as tf  # type: ignore[import]
        tflite = tf.lite  # type: ignore[assignment]
        _TFLITE_BACKEND = "tensorflow"
        _TFLITE_AVAILABLE = True
        logger.info("MobileFaceNet: Using tensorflow.lite interpreter backend.")
    except ImportError:
        tflite = None  # type: ignore[assignment]
        _TFLITE_BACKEND = "unavailable"
        _TFLITE_AVAILABLE = False
        logger.critical(
            "MobileFaceNet: Neither tflite_runtime nor tensorflow is installed. "
            "Face authentication will be unavailable."
        )

try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning(
        "MobileFaceNet: OpenCV (cv2) not found. "
        "PIL will be used for image resizing if available."
    )

try:
    from PIL import Image as PILImage  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected spatial input dimensions for MobileFaceNet (height, width).
MOBILEFACENET_INPUT_SIZE: Tuple[int, int] = (112, 112)

#: Default pixel normalisation range → values mapped to [-1, 1].
PIXEL_MEAN: float = 127.5
PIXEL_STD: float = 128.0

#: Default cosine similarity threshold for positive face match.
DEFAULT_VERIFICATION_THRESHOLD: float = 0.65

#: Minimum acceptable batch size.
MIN_BATCH_SIZE: int = 1

#: Maximum supported batch size for ``extract_embeddings_batch``.
MAX_BATCH_SIZE: int = 16

#: Absolute path to the TFLite model file.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # …/Tata/
DEFAULT_MODEL_PATH: Path = (
    _REPO_ROOT / "backend" / "ml" / "models_saved" / "mobilefacenet.tflite"
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FaceVerificationResult:
    """Result produced by :meth:`MobileFaceNetModel.verify_faces`.

    Attributes:
        verified: True if the two embeddings exceed the similarity threshold.
        similarity: Cosine similarity in the range [-1.0, 1.0].
        threshold: The threshold value used for this comparison.
        inference_ms: Wall-clock inference latency in milliseconds (0 if not measured).
    """

    verified: bool
    similarity: float
    threshold: float
    inference_ms: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Serialises the result to a JSON-serialisable dictionary.

        Returns:
            Dict[str, object]: Plain dictionary with all result fields.
        """
        return {
            "verified": self.verified,
            "similarity": round(self.similarity, 6),
            "threshold": self.threshold,
            "inference_ms": round(self.inference_ms, 3),
        }


# ---------------------------------------------------------------------------
# MobileFaceNet Model
# ---------------------------------------------------------------------------


class MobileFaceNetModel:
    """Thread-safe singleton wrapper around the MobileFaceNet TFLite model.

    Provides a complete face embedding pipeline from raw RGB crops to
    L2-normalized 128-D/512-D vectors suitable for cosine similarity matching.

    The singleton is lazily instantiated on first call to :meth:`get_instance`.
    All inference methods acquire the internal ``_infer_lock`` to prevent
    concurrent TFLite interpreter invocations.

    Attributes:
        _instance: Class-level singleton reference (``None`` until first call).
        _class_lock: Class-level threading lock guarding singleton creation.
        _infer_lock: Instance-level threading lock guarding TFLite inference.
        _interpreter: The loaded ``tflite.Interpreter`` instance or ``None``.
        _input_details: TFLite input tensor metadata list.
        _output_details: TFLite output tensor metadata list.
        _model_loaded: True when the interpreter is successfully initialised.
        _embedding_dim: Embedding vector dimensionality inferred from the model.
        _input_shape: Full input tensor shape including batch dimension.
        _model_path: Resolved path to the ``.tflite`` model file.
        _verification_threshold: Configurable cosine similarity threshold.
        _warmup_done: True after a successful warm-up pass has been completed.
    """

    _instance: Optional["MobileFaceNetModel"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        num_threads: int = 4,
    ) -> None:
        """Initialises the MobileFaceNetModel and attempts to load the TFLite model.

        Args:
            model_path: Filesystem path to the ``mobilefacenet.tflite`` file.
            verification_threshold: Cosine similarity cut-off for positive match.
                Values in the range [0.5, 0.9] are typical; 0.65 is a safe default.
            num_threads: Number of CPU threads allocated to the TFLite interpreter.
                For automotive edge hardware 4 threads is a reasonable default.

        Note:
            If the model file is not found or TFLite is unavailable, the instance
            is still created but ``_model_loaded`` is ``False``. All inference
            methods will raise ``RuntimeError`` in that state. This prevents a
            hard crash at import time and allows health-check endpoints to report
            the degraded state correctly.
        """
        self._model_path: Path = Path(model_path).resolve()
        self._verification_threshold: float = float(verification_threshold)
        self._num_threads: int = max(1, num_threads)

        # Interpreter state
        self._interpreter: Optional[object] = None  # tflite.Interpreter
        self._input_details: List[Dict] = []
        self._output_details: List[Dict] = []
        self._model_loaded: bool = False
        self._embedding_dim: int = 0
        self._input_shape: Tuple[int, ...] = ()

        # Thread-safety
        self._infer_lock: threading.Lock = threading.Lock()

        # Performance tracking
        self._warmup_done: bool = False
        self._total_inferences: int = 0
        self._total_inference_ms: float = 0.0

        # Attempt model load
        self._load_model()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        num_threads: int = 4,
    ) -> "MobileFaceNetModel":
        """Returns the shared :class:`MobileFaceNetModel` singleton.

        Thread-safe double-checked locking pattern. On first call the model
        is loaded from disk. Subsequent calls return the cached instance
        without I/O.

        Args:
            model_path: Forwarded to ``__init__`` on first call only.
            verification_threshold: Forwarded to ``__init__`` on first call only.
            num_threads: CPU thread count for TFLite; forwarded on first call.

        Returns:
            MobileFaceNetModel: The application-wide singleton instance.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    logger.info(
                        "MobileFaceNetModel: Creating singleton instance from %s",
                        model_path,
                    )
                    cls._instance = cls(
                        model_path=model_path,
                        verification_threshold=verification_threshold,
                        num_threads=num_threads,
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Public method to (re-)load the TFLite model from disk.

        Useful for hot-reloading after the model file has been updated.
        Acquires the inference lock to prevent concurrent invocation.

        Returns:
            bool: True if the model loaded successfully, False otherwise.
        """
        with self._infer_lock:
            return self._load_model()

    def _load_model(self) -> bool:
        """Internal model loading implementation (not lock-protected).

        Attempts to instantiate a TFLite Interpreter, allocate tensors, and
        read input/output metadata. Populates ``_embedding_dim`` and
        ``_input_shape`` from the model graph.

        Returns:
            bool: True on success, False on any failure.
        """
        if not _TFLITE_AVAILABLE:
            logger.error(
                "MobileFaceNetModel: TFLite backend unavailable. "
                "Install tflite_runtime or tensorflow."
            )
            return False

        if not self._model_path.exists():
            logger.error(
                "MobileFaceNetModel: Model file not found at '%s'. "
                "Place mobilefacenet.tflite in backend/ml/models_saved/.",
                self._model_path,
            )
            return False

        try:
            logger.info(
                "MobileFaceNetModel: Loading TFLite model from '%s' "
                "(%.1f MB) with %d thread(s).",
                self._model_path,
                self._model_path.stat().st_size / 1_048_576,
                self._num_threads,
            )

            # Instantiate interpreter
            if _TFLITE_BACKEND == "tflite_runtime":
                interpreter = tflite.Interpreter(  # type: ignore[call-arg]
                    model_path=str(self._model_path),
                    num_threads=self._num_threads,
                )
            else:
                # tensorflow.lite.Interpreter
                interpreter = tflite.Interpreter(  # type: ignore[call-arg]
                    model_path=str(self._model_path),
                    num_threads=self._num_threads,
                )

            interpreter.allocate_tensors()

            # Read tensor metadata
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            if not input_details or not output_details:
                logger.error(
                    "MobileFaceNetModel: No input/output tensors found in the model graph."
                )
                return False

            # Store state
            self._interpreter = interpreter
            self._input_details = input_details
            self._output_details = output_details

            # Resolve shapes
            self._input_shape = tuple(input_details[0]["shape"])  # e.g. (1, 112, 112, 3)
            output_shape = tuple(output_details[0]["shape"])       # e.g. (1, 128) or (1, 512)
            self._embedding_dim = int(output_shape[-1])

            self._model_loaded = True

            logger.info(
                "MobileFaceNetModel: ✓ Loaded — input_shape=%s | "
                "embedding_dim=%d | threshold=%.2f",
                self._input_shape,
                self._embedding_dim,
                self._verification_threshold,
            )
            return True

        except Exception as exc:
            logger.exception(
                "MobileFaceNetModel: Failed to load model from '%s': %s",
                self._model_path,
                exc,
            )
            self._model_loaded = False
            return False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocesses a raw face crop into a model-ready input tensor.

        The pipeline performs the following steps in order:
            1. **BGR→RGB conversion** (if cv2 is used and image appears BGR).
            2. **Resize** to ``MOBILEFACENET_INPUT_SIZE`` (112×112) using bilinear
               interpolation.
            3. **Cast** to ``float32``.
            4. **Pixel normalisation** to the range ``[-1, 1]`` via
               ``(pixel - 127.5) / 128.0``.
            5. **Batch dimension expansion** → shape ``(1, 112, 112, 3)``.

        Args:
            face_image: An RGB (preferred) or BGR face crop as a ``numpy.ndarray``
                of shape ``(H, W, 3)``, dtype ``uint8`` or ``float32``.
                The image must already be cropped and aligned — no face detection
                is performed here.

        Returns:
            np.ndarray: A float32 tensor of shape ``(1, 112, 112, 3)`` normalised
                to ``[-1, 1]``, ready for direct insertion into the TFLite input tensor.

        Raises:
            ValueError: If the image is ``None``, not 3-channel, or has fewer
                than 4 pixels in any spatial dimension.
        """
        if face_image is None:
            raise ValueError("MobileFaceNetModel.preprocess_face: received None image.")

        if face_image.ndim != 3 or face_image.shape[2] != 3:
            raise ValueError(
                f"MobileFaceNetModel.preprocess_face: expected H×W×3 image, "
                f"got shape {face_image.shape}."
            )

        h, w = face_image.shape[:2]
        if h < 4 or w < 4:
            raise ValueError(
                f"MobileFaceNetModel.preprocess_face: image too small ({h}×{w}). "
                "Ensure the face crop is at least 4×4 pixels."
            )

        target_h, target_w = MOBILEFACENET_INPUT_SIZE

        # --- Step 1: Resize ---
        resized = self._resize_image(face_image, target_h, target_w)

        # --- Step 2: Cast to float32 ---
        img_f32 = resized.astype(np.float32)

        # --- Step 3: Normalise pixels to [-1, 1] ---
        img_f32 = (img_f32 - PIXEL_MEAN) / PIXEL_STD

        # --- Step 4: Validate shape ---
        assert img_f32.shape == (target_h, target_w, 3), (
            f"Unexpected shape after preprocessing: {img_f32.shape}"
        )

        # --- Step 5: Add batch dimension ---
        tensor = np.expand_dims(img_f32, axis=0)  # → (1, 112, 112, 3)

        return tensor

    @staticmethod
    def _resize_image(
        image: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Resizes an image using the best available backend.

        Priority: OpenCV > PIL > NumPy nearest-neighbor fallback.

        Args:
            image: Input image array of shape (H, W, 3).
            target_h: Target height in pixels.
            target_w: Target width in pixels.

        Returns:
            np.ndarray: Resized image of shape (target_h, target_w, 3).
        """
        if _CV2_AVAILABLE:
            return cv2.resize(
                image,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        if _PIL_AVAILABLE:
            pil_img = PILImage.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((target_w, target_h), PILImage.BILINEAR)
            return np.array(pil_img, dtype=image.dtype)

        # Fallback: nearest-neighbor resize via NumPy index broadcasting
        logger.warning(
            "MobileFaceNetModel: No cv2/PIL available — using NumPy nearest-neighbor resize."
        )
        row_indices = np.linspace(0, image.shape[0] - 1, target_h).astype(np.int32)
        col_indices = np.linspace(0, image.shape[1] - 1, target_w).astype(np.int32)
        return image[np.ix_(row_indices, col_indices)]

    # ------------------------------------------------------------------
    # Embedding Extraction
    # ------------------------------------------------------------------

    def extract_embedding(
        self,
        face_image: np.ndarray,
        *,
        already_preprocessed: bool = False,
    ) -> np.ndarray:
        """Extracts a single L2-normalized face embedding from a face crop.

        This is the primary single-image inference entry point. It is thread-safe
        via the internal ``_infer_lock``.

        Args:
            face_image: An RGB face crop array of shape ``(H, W, 3)`` or, if
                ``already_preprocessed=True``, a float32 tensor of shape
                ``(1, 112, 112, 3)``.
            already_preprocessed: Skip the preprocessing step if the caller has
                already produced a correctly shaped/normalised tensor. Defaults
                to ``False``.

        Returns:
            np.ndarray: An L2-normalized float32 embedding vector of shape ``(D,)``
                where ``D`` is the model's output dimension (typically 128 or 512).

        Raises:
            RuntimeError: If the model has not been loaded successfully.
            ValueError: If the preprocessed input has an unexpected shape.
        """
        self._assert_model_loaded()

        if already_preprocessed:
            tensor = face_image.astype(np.float32)
        else:
            tensor = self.preprocess_face(face_image)

        self._validate_input_tensor(tensor)

        with self._infer_lock:
            t0 = time.perf_counter()
            raw_embedding = self._run_inference(tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update stats
        self._total_inferences += 1
        self._total_inference_ms += elapsed_ms

        if elapsed_ms > 50.0:
            logger.warning(
                "MobileFaceNetModel: Slow inference detected — %.1f ms "
                "(target <20 ms). Consider reducing image resolution or "
                "increasing num_threads.",
                elapsed_ms,
            )
        else:
            logger.debug(
                "MobileFaceNetModel: Inference completed in %.2f ms.", elapsed_ms
            )

        normalized = self.normalize_embedding(raw_embedding)
        return normalized

    def extract_embeddings_batch(
        self,
        face_images: List[np.ndarray],
        *,
        already_preprocessed: bool = False,
    ) -> List[np.ndarray]:
        """Extracts L2-normalized embeddings for a batch of face crops.

        Processes images one-at-a-time through the TFLite interpreter (which is
        optimised for single-sample inference on CPU). For true batch parallelism,
        use a batch-capable ONNX/TF SavedModel instead.

        Args:
            face_images: A list of RGB face crop arrays, each of shape ``(H, W, 3)``.
                Must contain between 1 and ``MAX_BATCH_SIZE`` (16) images.
            already_preprocessed: If True, each element is assumed to already be
                a ``(1, 112, 112, 3)`` float32 tensor. Defaults to ``False``.

        Returns:
            List[np.ndarray]: A list of L2-normalized float32 embedding vectors,
                one per input image, each of shape ``(D,)``.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the batch is empty or exceeds ``MAX_BATCH_SIZE``.
        """
        self._assert_model_loaded()

        if not face_images:
            raise ValueError(
                "MobileFaceNetModel.extract_embeddings_batch: received empty list."
            )

        n = len(face_images)
        if n > MAX_BATCH_SIZE:
            raise ValueError(
                f"MobileFaceNetModel.extract_embeddings_batch: batch size {n} "
                f"exceeds maximum of {MAX_BATCH_SIZE}. Split into smaller chunks."
            )

        logger.debug(
            "MobileFaceNetModel: Processing batch of %d face(s).", n
        )

        embeddings: List[np.ndarray] = []
        for idx, img in enumerate(face_images):
            try:
                emb = self.extract_embedding(img, already_preprocessed=already_preprocessed)
                embeddings.append(emb)
            except Exception as exc:
                logger.error(
                    "MobileFaceNetModel.extract_embeddings_batch: "
                    "Failed on image %d/%d — %s. Inserting zero vector.",
                    idx + 1,
                    n,
                    exc,
                )
                embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))

        return embeddings

    def _run_inference(self, tensor: np.ndarray) -> np.ndarray:
        """Executes a single forward pass through the TFLite interpreter.

        **Must be called while holding ``_infer_lock``.**

        Args:
            tensor: Float32 tensor of shape matching the model's input spec.

        Returns:
            np.ndarray: Raw output embedding (not L2-normalized), shape ``(D,)``.
        """
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        self._interpreter.set_tensor(input_index, tensor)  # type: ignore[union-attr]
        self._interpreter.invoke()  # type: ignore[union-attr]

        raw: np.ndarray = self._interpreter.get_tensor(output_index)  # type: ignore[union-attr]

        # Squeeze batch dimension: (1, D) → (D,)
        return raw.squeeze(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisation & Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Applies L2 normalisation to project an embedding onto the unit hypersphere.

        After normalisation, cosine similarity between two vectors equals their
        dot product, enabling O(1) verification with a simple ``np.dot`` call.

        Args:
            embedding: A 1-D float32 numpy array of arbitrary L2 norm.

        Returns:
            np.ndarray: The unit-norm version of ``embedding``, dtype float32.
                If the input norm is below 1e-8 (near-zero vector), a zero vector
                is returned to avoid division-by-zero.
        """
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            logger.warning(
                "MobileFaceNetModel.normalize_embedding: "
                "Near-zero embedding vector detected (norm=%.2e). "
                "Returning zero vector.",
                norm,
            )
            return np.zeros_like(embedding, dtype=np.float32)
        return (embedding / norm).astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Computes the cosine similarity between two embedding vectors.

        For L2-normalized inputs (unit vectors), this is equivalent to the
        dot product and lies in the range ``[-1, 1]``. A value of ``1.0``
        means identical direction; ``-1.0`` means maximally dissimilar;
        ``0.0`` means orthogonal / unrelated.

        Args:
            a: First embedding vector, shape ``(D,)``, float32.
            b: Second embedding vector, shape ``(D,)``, float32.

        Returns:
            float: Cosine similarity in ``[-1.0, 1.0]``.

        Note:
            If either vector has near-zero norm the function returns ``0.0``
            to prevent division-by-zero artefacts.
        """
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a < 1e-8 or norm_b < 1e-8:
            logger.warning(
                "MobileFaceNetModel.cosine_similarity: "
                "One or both vectors have near-zero norm. Returning 0.0."
            )
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to valid range to handle floating-point overflow edge cases
        return float(np.clip(similarity, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_faces(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> FaceVerificationResult:
        """Determines whether two face embeddings belong to the same person.

        Uses cosine similarity with a configurable threshold. This method
        operates directly on pre-extracted embeddings and is therefore
        extremely fast (O(D) dot product).

        Args:
            embedding_a: L2-normalized face embedding, shape ``(D,)``.
            embedding_b: L2-normalized face embedding, shape ``(D,)``.
            threshold: Override the instance ``_verification_threshold`` for
                this call only. If ``None``, the default threshold is used.

        Returns:
            FaceVerificationResult: A dataclass with:
                - ``verified`` (bool): True if similarity ≥ threshold.
                - ``similarity`` (float): Cosine similarity score.
                - ``threshold`` (float): The threshold used for this decision.
                - ``inference_ms`` (float): Always 0.0 (embeddings pre-computed).

        Example::

            emb_enrolled = model.extract_embedding(enrolled_face_crop)
            emb_live = model.extract_embedding(live_camera_crop)
            result = model.verify_faces(emb_enrolled, emb_live)
            if result.verified:
                print(f"Driver authenticated (similarity={result.similarity:.4f})")
        """
        effective_threshold = (
            threshold if threshold is not None else self._verification_threshold
        )

        similarity = self.cosine_similarity(embedding_a, embedding_b)
        verified = similarity >= effective_threshold

        logger.info(
            "MobileFaceNetModel.verify_faces: similarity=%.4f | "
            "threshold=%.2f | verified=%s",
            similarity,
            effective_threshold,
            verified,
        )

        return FaceVerificationResult(
            verified=verified,
            similarity=similarity,
            threshold=effective_threshold,
            inference_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warmup(self, num_runs: int = 3) -> float:
        """Runs several dummy inference passes to prime the TFLite interpreter.

        JIT compilation and kernel loading in TFLite can cause the first
        inference to be significantly slower than subsequent ones. Calling
        ``warmup()`` at application startup amortises this latency.

        Args:
            num_runs: Number of dummy forward passes to execute. Defaults to 3.

        Returns:
            float: Average inference latency in milliseconds across all warm-up
                runs. Returns ``-1.0`` if the model is not loaded.

        Example::

            model = MobileFaceNetModel.get_instance()
            avg_ms = model.warmup()
            logger.info("MobileFaceNet warm-up complete: avg=%.1f ms", avg_ms)
        """
        if not self._model_loaded:
            logger.warning(
                "MobileFaceNetModel.warmup: Model not loaded — skipping warm-up."
            )
            return -1.0

        logger.info(
            "MobileFaceNetModel.warmup: Starting %d warm-up inference run(s).", num_runs
        )

        # Create a random dummy input tensor
        h, w = MOBILEFACENET_INPUT_SIZE
        dummy_input = np.random.uniform(-1.0, 1.0, (1, h, w, 3)).astype(np.float32)

        latencies: List[float] = []
        for run_idx in range(num_runs):
            try:
                with self._infer_lock:
                    t0 = time.perf_counter()
                    _ = self._run_inference(dummy_input)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                logger.debug(
                    "MobileFaceNetModel.warmup: Run %d/% d → %.2f ms",
                    run_idx + 1,
                    num_runs,
                    elapsed_ms,
                )
            except Exception as exc:
                logger.error(
                    "MobileFaceNetModel.warmup: Run %d failed — %s", run_idx + 1, exc
                )

        if not latencies:
            return -1.0

        avg_ms = float(np.mean(latencies))
        self._warmup_done = True

        logger.info(
            "MobileFaceNetModel.warmup: ✓ Complete — "
            "avg=%.2f ms | min=%.2f ms | max=%.2f ms",
            avg_ms,
            min(latencies),
            max(latencies),
        )
        return avg_ms

    # ------------------------------------------------------------------
    # Metadata / Properties
    # ------------------------------------------------------------------

    def get_embedding_dimension(self) -> int:
        """Returns the output embedding dimension of the loaded model.

        Returns:
            int: Number of dimensions in the embedding vector (e.g. 128 or 512).
                Returns ``0`` if the model has not been loaded.
        """
        return self._embedding_dim

    @property
    def model_loaded(self) -> bool:
        """True if the TFLite interpreter is successfully initialised."""
        return self._model_loaded

    @property
    def verification_threshold(self) -> float:
        """The default cosine similarity threshold for positive face verification."""
        return self._verification_threshold

    @verification_threshold.setter
    def verification_threshold(self, value: float) -> None:
        """Sets a new verification threshold.

        Args:
            value: New threshold value. Must be in the range (0, 1].

        Raises:
            ValueError: If value is outside the valid range.
        """
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"verification_threshold must be in (0, 1]. Got: {value}"
            )
        logger.info(
            "MobileFaceNetModel: Updating verification threshold "
            "%.2f → %.2f",
            self._verification_threshold,
            value,
        )
        self._verification_threshold = value

    @property
    def input_size(self) -> Tuple[int, int]:
        """Expected spatial input dimensions ``(height, width)``."""
        return MOBILEFACENET_INPUT_SIZE

    @property
    def average_inference_ms(self) -> float:
        """Running average inference latency in milliseconds.

        Returns ``0.0`` if no inferences have been performed yet.
        """
        if self._total_inferences == 0:
            return 0.0
        return self._total_inference_ms / self._total_inferences

    @property
    def tflite_backend(self) -> str:
        """Name of the active TFLite backend (``tflite_runtime`` or ``tensorflow``)."""
        return _TFLITE_BACKEND

    def get_model_info(self) -> Dict[str, object]:
        """Returns a structured summary of the model's operational state.

        Returns:
            Dict[str, object]: Dictionary with keys:
                - ``model_loaded`` (bool)
                - ``model_path`` (str)
                - ``embedding_dimension`` (int)
                - ``input_shape`` (tuple)
                - ``verification_threshold`` (float)
                - ``warmup_done`` (bool)
                - ``tflite_backend`` (str)
                - ``average_inference_ms`` (float)
                - ``total_inferences`` (int)
        """
        return {
            "model_loaded": self._model_loaded,
            "model_path": str(self._model_path),
            "embedding_dimension": self._embedding_dim,
            "input_shape": self._input_shape,
            "verification_threshold": self._verification_threshold,
            "warmup_done": self._warmup_done,
            "tflite_backend": _TFLITE_BACKEND,
            "average_inference_ms": round(self.average_inference_ms, 3),
            "total_inferences": self._total_inferences,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_model_loaded(self) -> None:
        """Raises RuntimeError if the model has not been loaded successfully.

        Raises:
            RuntimeError: If ``_model_loaded`` is False.
        """
        if not self._model_loaded:
            raise RuntimeError(
                "MobileFaceNetModel: TFLite model is not loaded. "
                f"Verify that the model file exists at '{self._model_path}' "
                "and that tflite_runtime or tensorflow is installed."
            )

    def _validate_input_tensor(self, tensor: np.ndarray) -> None:
        """Validates that the preprocessed tensor matches the expected input spec.

        Args:
            tensor: Preprocessed float32 input tensor.

        Raises:
            ValueError: If shape or dtype do not match the model's expectations.
        """
        expected_shape = tuple(self._input_details[0]["shape"])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MobileFaceNetModel: Input tensor shape mismatch. "
                f"Expected {expected_shape}, got {tuple(tensor.shape)}."
            )
        if tensor.dtype != np.float32:
            raise ValueError(
                f"MobileFaceNetModel: Input tensor dtype must be float32, "
                f"got {tensor.dtype}."
            )

    def __repr__(self) -> str:
        return (
            f"MobileFaceNetModel("
            f"loaded={self._model_loaded}, "
            f"embedding_dim={self._embedding_dim}, "
            f"threshold={self._verification_threshold:.2f}, "
            f"backend={_TFLITE_BACKEND})"
        )
