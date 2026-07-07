"""MobileFaceNet model wrapper using TensorFlow Lite for facial embeddings."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("CogniDrive.MobileFaceNet")

# Handle dynamic tflite imports
try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_BACKEND = "tflite_runtime"
    _TFLITE_AVAILABLE = True
    logger.info("MobileFaceNet: Using tflite_runtime interpreter backend.")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        _TFLITE_BACKEND = "tensorflow"
        _TFLITE_AVAILABLE = True
        logger.info("MobileFaceNet: Using tensorflow.lite interpreter backend.")
    except ImportError:
        tflite = None
        _TFLITE_BACKEND = "unavailable"
        _TFLITE_AVAILABLE = False
        logger.critical("MobileFaceNet: Neither tflite_runtime nor tensorflow is installed.")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("MobileFaceNet: OpenCV not found. Falling back to PIL or NumPy.")

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

MOBILEFACENET_INPUT_SIZE: Tuple[int, int] = (112, 112)
PIXEL_MEAN: float = 127.5
PIXEL_STD: float = 128.0
DEFAULT_VERIFICATION_THRESHOLD: float = 0.65
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 16

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH: Path = _REPO_ROOT / "backend" / "ml" / "models_saved" / "mobilefacenet.tflite"


@dataclass
class FaceVerificationResult:
    """Verification output comparison metrics."""
    verified: bool
    similarity: float
    threshold: float
    inference_ms: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "verified": self.verified,
            "similarity": round(self.similarity, 6),
            "threshold": self.threshold,
            "inference_ms": round(self.inference_ms, 3),
        }


class MobileFaceNetModel:
    """Wrapper around MobileFaceNet TFLite for facial recognition/embeddings."""

    _instance: Optional["MobileFaceNetModel"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        num_threads: int = 4,
    ) -> None:
        self._model_path = Path(model_path).resolve()
        self._verification_threshold = float(verification_threshold)
        self._num_threads = max(1, num_threads)

        self._interpreter = None
        self._input_details: List[Dict] = []
        self._output_details: List[Dict] = []
        self._model_loaded = False
        self._embedding_dim = 0
        self._input_shape: Tuple[int, ...] = ()

        self._infer_lock = threading.Lock()
        self._warmup_done = False
        self._total_inferences = 0
        self._total_inference_ms = 0.0

        self._load_model()

    @classmethod
    def get_instance(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        num_threads: int = 4,
    ) -> "MobileFaceNetModel":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_path=model_path,
                        verification_threshold=verification_threshold,
                        num_threads=num_threads,
                    )
        return cls._instance

    def load_model(self) -> bool:
        with self._infer_lock:
            return self._load_model()

    def _load_model(self) -> bool:
        if not _TFLITE_AVAILABLE:
            logger.error("MobileFaceNetModel: TFLite interpreter not installed.")
            return False

        if not self._model_path.exists():
            logger.error("MobileFaceNetModel: Model not found at %s", self._model_path)
            return False

        try:
            logger.info("Loading TFLite model from %s", self._model_path)
            self._interpreter = tflite.Interpreter(
                model_path=str(self._model_path),
                num_threads=self._num_threads,
            )
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            if not self._input_details or not self._output_details:
                logger.error("MobileFaceNetModel: Invalid input/output tensors.")
                return False

            self._input_shape = tuple(self._input_details[0]["shape"])
            self._embedding_dim = int(self._output_details[0]["shape"][-1])
            self._model_loaded = True
            return True

        except Exception as exc:
            logger.exception("Failed to load TFLite interpreter: %s", exc)
            self._model_loaded = False
            return False

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        if face_image is None:
            raise ValueError("preprocess_face: received None image.")
        if face_image.ndim != 3 or face_image.shape[2] != 3:
            raise ValueError(f"preprocess_face: expected HxWx3 image, got shape {face_image.shape}")

        h, w = face_image.shape[:2]
        if h < 4 or w < 4:
            raise ValueError(f"preprocess_face: image too small ({h}x{w})")

        target_h, target_w = MOBILEFACENET_INPUT_SIZE
        resized = self._resize_image(face_image, target_h, target_w)
        img_f32 = resized.astype(np.float32)
        img_f32 = (img_f32 - PIXEL_MEAN) / PIXEL_STD
        return np.expand_dims(img_f32, axis=0)

    @staticmethod
    def _resize_image(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if _CV2_AVAILABLE:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if _PIL_AVAILABLE:
            pil_img = PILImage.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((target_w, target_h), PILImage.BILINEAR)
            return np.array(pil_img, dtype=image.dtype)

        # Basic NumPy fallback if nothing else is available
        logger.warning("No resize backend available, using NumPy interpolation.")
        row_indices = np.linspace(0, image.shape[0] - 1, target_h).astype(np.int32)
        col_indices = np.linspace(0, image.shape[1] - 1, target_w).astype(np.int32)
        return image[np.ix_(row_indices, col_indices)]

    def extract_embedding(
        self,
        face_image: np.ndarray,
        *,
        already_preprocessed: bool = False,
    ) -> np.ndarray:
        self._assert_model_loaded()

        tensor = face_image.astype(np.float32) if already_preprocessed else self.preprocess_face(face_image)
        self._validate_input_tensor(tensor)

        with self._infer_lock:
            t0 = time.perf_counter()
            raw_embedding = self._run_inference(tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._total_inferences += 1
        self._total_inference_ms += elapsed_ms

        if elapsed_ms > 50.0:
            logger.warning("Slow model inference: %.1f ms", elapsed_ms)

        return self.normalize_embedding(raw_embedding)

    def extract_embeddings_batch(
        self,
        face_images: List[np.ndarray],
        *,
        already_preprocessed: bool = False,
    ) -> List[np.ndarray]:
        self._assert_model_loaded()

        if not face_images:
            raise ValueError("extract_embeddings_batch: received empty list.")

        n = len(face_images)
        if n > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {n} exceeds limit of {MAX_BATCH_SIZE}.")

        embeddings: List[np.ndarray] = []
        for idx, img in enumerate(face_images):
            try:
                emb = self.extract_embedding(img, already_preprocessed=already_preprocessed)
                embeddings.append(emb)
            except Exception as exc:
                logger.error("Failed to process batch image %d: %s. Using zeros.", idx, exc)
                embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))

        return embeddings

    def _run_inference(self, tensor: np.ndarray) -> np.ndarray:
        input_index = self._input_details[0]["index"]
        output_index = self._output_details[0]["index"]

        self._interpreter.set_tensor(input_index, tensor)
        self._interpreter.invoke()

        raw = self._interpreter.get_tensor(output_index)
        return raw.squeeze(axis=0).astype(np.float32)

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            logger.warning("Zero norm embedding vector detected.")
            return np.zeros_like(embedding, dtype=np.float32)
        return (embedding / norm).astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(similarity, -1.0, 1.0))

    def verify_faces(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> FaceVerificationResult:
        val_thresh = threshold if threshold is not None else self._verification_threshold
        similarity = self.cosine_similarity(embedding_a, embedding_b)
        verified = similarity >= val_thresh

        logger.info("Verify: sim=%.4f, thresh=%.2f, verified=%s", similarity, val_thresh, verified)
        return FaceVerificationResult(verified=verified, similarity=similarity, threshold=val_thresh)

    def warmup(self, num_runs: int = 3) -> float:
        if not self._model_loaded:
            logger.warning("Model not loaded - skipping warmup.")
            return -1.0

        logger.info("Running %d warm-up inference cycles...", num_runs)
        h, w = MOBILEFACENET_INPUT_SIZE
        dummy_input = np.random.uniform(-1.0, 1.0, (1, h, w, 3)).astype(np.float32)

        latencies: List[float] = []
        for _ in range(num_runs):
            try:
                with self._infer_lock:
                    t0 = time.perf_counter()
                    self._run_inference(dummy_input)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                logger.error("Warmup run cycle failed: %s", exc)

        if not latencies:
            return -1.0

        avg_ms = float(np.mean(latencies))
        self._warmup_done = True
        return avg_ms

    def get_embedding_dimension(self) -> int:
        return self._embedding_dim

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def verification_threshold(self) -> float:
        return self._verification_threshold

    @verification_threshold.setter
    def verification_threshold(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Threshold must be in (0, 1]. Got: {value}")
        self._verification_threshold = value

    @property
    def input_size(self) -> Tuple[int, int]:
        return MOBILEFACENET_INPUT_SIZE

    @property
    def average_inference_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return self._total_inference_ms / self._total_inferences

    @property
    def tflite_backend(self) -> str:
        return _TFLITE_BACKEND

    def get_model_info(self) -> Dict[str, object]:
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

    def _assert_model_loaded(self) -> None:
        if not self._model_loaded:
            raise RuntimeError(f"TFLite model not loaded from {self._model_path}")

    def _validate_input_tensor(self, tensor: np.ndarray) -> None:
        expected_shape = tuple(self._input_details[0]["shape"])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Input shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != np.float32:
            raise ValueError(f"Input dtype must be float32, got {tensor.dtype}")

    def __repr__(self) -> str:
        return f"MobileFaceNetModel(loaded={self._model_loaded}, dim={self._embedding_dim}, backend={_TFLITE_BACKEND})"
