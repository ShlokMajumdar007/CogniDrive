"""Driver Digital Twin — threshold manager.

This module is the authoritative, persisted source of truth for every
driver's personalised thresholds. While
:mod:`backend.ml.digital_twin.personalization` *computes* an adaptive
threshold on demand from a live in-memory baseline,
:class:`ThresholdManager` is responsible for:

    1. **Maintaining** the current personalised threshold snapshot per
       driver (a simple, fast-lookup cache independent of whether the
       driver is currently in an active session).
    2. **Providing fallback defaults** when no personalised snapshot exists
       yet (cold-start, or after a cache eviction / cold service restart).
    3. **Handling threshold drift** — detecting when a freshly computed
       threshold has moved suspiciously far from its last persisted value
       (e.g. due to a temporary tracking glitch flooding the baseline with
       bad samples) and damping the update rather than applying it outright.
    4. **Persisting thresholds** to local disk as JSON, fully offline, so
       personalisation survives process restarts without requiring a
       database schema migration (the current
       :class:`backend.database.models.driver_profile.DriverProfile` ORM
       model has no generic JSON/blob column for arbitrary threshold data).

Position in the pipeline::

    PersonalizationEngine.compute_all_thresholds(driver_id)
        --> ThresholdManager.update(driver_id, thresholds)   # drift check + cache + persist
        --> ThresholdManager.get(driver_id, signal)           # fast lookup at inference time

Persistence format
-------------------
Each driver's thresholds are stored as a single JSON file at
``<storage_dir>/driver_<driver_id>_thresholds.json``, keeping the on-disk
layout under the project's existing ``datasets/driver_profiles/`` directory
and requiring zero database changes. This keeps the module fully
self-contained and trivially inspectable/debuggable from the command line.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from backend.digital_twin.personalization import (
        AdaptiveThreshold,
        TrackedSignal,
        _global_defaults,
    )
except ImportError:
    from digital_twin.personalization import (  # type: ignore[no-redef]
        AdaptiveThreshold,
        TrackedSignal,
        _global_defaults,
    )



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory for on-disk threshold persistence, relative to the
#: project root, matching the `datasets/driver_profiles/` path declared in
#: the project's production folder structure.
DEFAULT_STORAGE_DIR: Path = Path("datasets/driver_profiles")

#: Maximum allowed fractional change between a driver's last-persisted
#: threshold value and a newly computed one before it is treated as
#: "drift" and damped rather than applied directly. E.g. 0.40 means a new
#: threshold more than 40% different from the last persisted value is
#: suspicious.
MAX_RELATIVE_DRIFT: float = 0.40

#: Damping factor applied to drift-flagged updates: the new cached value
#: moves only this fraction of the way from the old value toward the newly
#: computed (drifted) value, rather than jumping directly to it.
DRIFT_DAMPING_FACTOR: float = 0.25

#: Numerical floor to avoid division-by-zero when computing relative drift
#: against a near-zero previous threshold.
_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Persisted threshold record
# ---------------------------------------------------------------------------


@dataclass
class ThresholdRecord:
    """A single signal's currently active, persisted threshold.

    Attributes:
        signal: Which tracked signal this threshold applies to.
        value: The currently active threshold value (post drift-damping).
        confidence: Personalisation confidence at the time this value was
            last updated, ``[0, 1]``.
        source: ``"personalized"`` if derived from a driver baseline (even
            if damped), or ``"default"`` if it is the raw global fallback.
        drift_flagged: True if the most recent update for this signal
            triggered drift damping.
        last_updated: UTC timestamp of the last update.
    """

    signal: TrackedSignal
    value: float
    confidence: float
    source: str
    drift_flagged: bool = False
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, object]:
        """Serialises this record to a JSON-compatible dictionary.

        Returns:
            Dict[str, object]: Flat dictionary representation.
        """
        return {
            "signal": self.signal.value,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "drift_flagged": self.drift_flagged,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ThresholdRecord":
        """Reconstructs a record from a previously serialised dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            ThresholdRecord: Reconstructed record.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If ``signal`` does not match a known
                :class:`TrackedSignal`.
        """
        return cls(
            signal=TrackedSignal(data["signal"]),
            value=float(data["value"]),  # type: ignore[arg-type]
            confidence=float(data["confidence"]),  # type: ignore[arg-type]
            source=str(data["source"]),
            drift_flagged=bool(data.get("drift_flagged", False)),
            last_updated=datetime.fromisoformat(str(data["last_updated"]))
            if "last_updated" in data
            else datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Threshold Manager
# ---------------------------------------------------------------------------


class ThresholdManager:
    """Maintains, persists, and serves each driver's authoritative threshold set.

    This is the production-facing API the inference engine and API layer
    should call — they should never need to talk to
    :class:`~backend.ml.digital_twin.personalization.PersonalizationEngine`
    directly for *reading* thresholds, only for the periodic *computation*
    step that feeds :meth:`update`.

    Attributes:
        storage_dir: Directory where per-driver threshold JSON files are
            written and read.
        max_relative_drift: Forwarded to drift detection (see module docstring).
        drift_damping_factor: Forwarded to drift damping.
    """

    def __init__(
        self,
        storage_dir: Path = DEFAULT_STORAGE_DIR,
        max_relative_drift: float = MAX_RELATIVE_DRIFT,
        drift_damping_factor: float = DRIFT_DAMPING_FACTOR,
    ) -> None:
        """Initialises the threshold manager.

        Args:
            storage_dir: Directory for on-disk per-driver threshold JSON
                files. Created automatically if it does not exist.
            max_relative_drift: Fractional change threshold above which an
                update is flagged as drift and damped (see
                :data:`MAX_RELATIVE_DRIFT`).
            drift_damping_factor: Interpolation factor applied to
                drift-flagged updates (see :data:`DRIFT_DAMPING_FACTOR`).

        Raises:
            ValueError: If ``max_relative_drift`` or ``drift_damping_factor``
                are not in ``(0, 1]``.
            RuntimeError: If ``storage_dir`` cannot be created.
        """
        if not (0.0 < max_relative_drift <= 1.0):
            raise ValueError(
                f"max_relative_drift must be in (0, 1], got {max_relative_drift}."
            )
        if not (0.0 < drift_damping_factor <= 1.0):
            raise ValueError(
                f"drift_damping_factor must be in (0, 1], got {drift_damping_factor}."
            )

        self.storage_dir = Path(storage_dir)
        self.max_relative_drift = max_relative_drift
        self.drift_damping_factor = drift_damping_factor

        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Failed to create threshold storage directory %s: %s",
                self.storage_dir,
                exc,
            )
            raise RuntimeError(
                f"Could not create threshold storage directory: {self.storage_dir}"
            ) from exc

        self._cache: Dict[int, Dict[TrackedSignal, ThresholdRecord]] = {}
        self._lock = threading.Lock()
        self._global_defaults = _global_defaults()

        logger.info(
            "ThresholdManager initialised (storage_dir=%s, max_relative_drift=%.2f, "
            "drift_damping_factor=%.2f).",
            self.storage_dir,
            self.max_relative_drift,
            self.drift_damping_factor,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _file_path(self, driver_id: int) -> Path:
        """Resolves the on-disk JSON path for a driver's threshold snapshot.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            Path: Resolved file path under :attr:`storage_dir`.
        """
        return self.storage_dir / f"driver_{driver_id}_thresholds.json"

    # ------------------------------------------------------------------
    # Fallback defaults
    # ------------------------------------------------------------------

    def get_default(self, signal: TrackedSignal) -> float:
        """Returns the global population-level default threshold for a signal.

        This is the unconditional fallback used whenever no personalised
        record exists (new driver, evicted cache, corrupted persistence
        file) — it can never fail or return ``None``.

        Args:
            signal: Which tracked signal to look up.

        Returns:
            float: The global default threshold value, sourced from
            :class:`backend.app.constants.Thresholds`.
        """
        return self._global_defaults[signal]

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def _detect_and_damp_drift(
        self, previous_value: Optional[float], new_value: float
    ) -> Tuple[float, bool]:
        """Detects threshold drift and applies damping if necessary.

        Args:
            previous_value: The last persisted/cached value for this
                signal, or ``None`` if there is no prior value (first-ever
                update — never flagged as drift).
            new_value: The freshly computed candidate threshold value.

        Returns:
            Tuple[float, bool]: ``(final_value, drift_flagged)``. If drift
            is detected, ``final_value`` is damped toward ``new_value``
            rather than equal to it; otherwise ``final_value == new_value``.
        """
        if previous_value is None:
            return new_value, False

        denom = abs(previous_value) + _EPS
        relative_change = abs(new_value - previous_value) / denom

        if relative_change <= self.max_relative_drift:
            return new_value, False

        damped_value = (
            previous_value
            + self.drift_damping_factor * (new_value - previous_value)
        )
        logger.warning(
            "Threshold drift detected: previous=%.6f, candidate=%.6f "
            "(relative_change=%.2f%% > %.2f%%); damping to %.6f.",
            previous_value,
            new_value,
            100.0 * relative_change,
            100.0 * self.max_relative_drift,
            damped_value,
        )
        return damped_value, True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self, driver_id: int, thresholds: Dict[TrackedSignal, AdaptiveThreshold]
    ) -> Dict[TrackedSignal, ThresholdRecord]:
        """Applies a freshly computed set of adaptive thresholds for a driver.

        For each signal: checks the new value against the last cached
        value for drift, applies damping if necessary, updates the
        in-memory cache, and persists the full per-driver snapshot to disk.

        Args:
            driver_id: Primary key of the driver profile.
            thresholds: Output of
                :meth:`backend.ml.digital_twin.personalization.PersonalizationEngine.compute_all_thresholds`
                (or a subset thereof) for this driver.

        Returns:
            Dict[TrackedSignal, ThresholdRecord]: The updated, authoritative
            threshold records now active for this driver.

        Raises:
            ValueError: If ``driver_id`` is not a positive integer.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            existing = self._cache.get(driver_id, {})
            updated: Dict[TrackedSignal, ThresholdRecord] = {}

            for signal, adaptive in thresholds.items():
                previous_record = existing.get(signal)
                previous_value = previous_record.value if previous_record is not None else None

                final_value, drift_flagged = self._detect_and_damp_drift(
                    previous_value, adaptive.value
                )

                source = "personalized" if adaptive.confidence > 0.0 else "default"

                updated[signal] = ThresholdRecord(
                    signal=signal,
                    value=final_value,
                    confidence=adaptive.confidence,
                    source=source,
                    drift_flagged=drift_flagged,
                    last_updated=datetime.now(timezone.utc),
                )

            # Preserve any signals not included in this update call.
            for signal, record in existing.items():
                if signal not in updated:
                    updated[signal] = record

            self._cache[driver_id] = updated

        self._persist_to_disk(driver_id, updated)

        logger.info(
            "Updated thresholds for driver_id=%d: %d signals (%d drift-flagged).",
            driver_id,
            len(updated),
            sum(1 for r in updated.values() if r.drift_flagged),
        )
        return updated

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get(self, driver_id: int, signal: TrackedSignal) -> float:
        """Returns the currently active threshold value for a driver and signal.

        Resolution order:
            1. In-memory cache (fastest path, populated by :meth:`update`
               or :meth:`load_from_disk`).
            2. On-disk persisted snapshot (auto-loaded into cache on miss).
            3. Global default (unconditional fallback, never fails).

        Args:
            driver_id: Primary key of the driver profile.
            signal: Which tracked signal to look up.

        Returns:
            float: The active threshold value to use for this
            driver/signal pair.

        Raises:
            ValueError: If ``driver_id`` is not a positive integer.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            cached = self._cache.get(driver_id)

        if cached is not None and signal in cached:
            return cached[signal].value

        # Cache miss — attempt to hydrate from disk before falling back.
        loaded = self.load_from_disk(driver_id)
        if loaded is not None and signal in loaded:
            return loaded[signal].value

        return self.get_default(signal)

    def get_record(self, driver_id: int, signal: TrackedSignal) -> ThresholdRecord:
        """Returns the full threshold record (value + provenance) for a driver/signal.

        Unlike :meth:`get`, this always returns a complete
        :class:`ThresholdRecord` (synthesising a ``"default"``-sourced one
        if nothing personalised exists), useful for explainability /
        dashboard display where confidence and source matter.

        Args:
            driver_id: Primary key of the driver profile.
            signal: Which tracked signal to look up.

        Returns:
            ThresholdRecord: The active record for this driver/signal.

        Raises:
            ValueError: If ``driver_id`` is not a positive integer.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            cached = self._cache.get(driver_id)

        if cached is not None and signal in cached:
            return cached[signal]

        loaded = self.load_from_disk(driver_id)
        if loaded is not None and signal in loaded:
            return loaded[signal]

        return ThresholdRecord(
            signal=signal,
            value=self.get_default(signal),
            confidence=0.0,
            source="default",
            drift_flagged=False,
        )

    def get_all(self, driver_id: int) -> Dict[TrackedSignal, ThresholdRecord]:
        """Returns the complete active threshold set for a driver.

        Any signal not present in cache/disk is synthesised from the
        global default, so the returned dictionary always has an entry for
        every :class:`TrackedSignal`.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            Dict[TrackedSignal, ThresholdRecord]: One record per tracked
            signal.

        Raises:
            ValueError: If ``driver_id`` is not a positive integer.
        """
        return {signal: self.get_record(driver_id, signal) for signal in TrackedSignal}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_to_disk(
        self, driver_id: int, records: Dict[TrackedSignal, ThresholdRecord]
    ) -> None:
        """Writes a driver's full threshold snapshot to its JSON file.

        Args:
            driver_id: Primary key of the driver profile.
            records: Complete set of threshold records to persist.

        Raises:
            RuntimeError: If the write fails (e.g. disk full, permissions).
        """
        path = self._file_path(driver_id)
        payload = {
            "driver_id": driver_id,
            "schema_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                signal.value: record.to_dict() for signal, record in records.items()
            },
        }

        tmp_path = path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp_path.replace(path)  # Atomic on POSIX filesystems.
        except OSError as exc:
            logger.error(
                "Failed to persist thresholds for driver_id=%d to %s: %s",
                driver_id,
                path,
                exc,
            )
            raise RuntimeError(
                f"Failed to write threshold snapshot for driver_id={driver_id}"
            ) from exc

    def load_from_disk(self, driver_id: int) -> Optional[Dict[TrackedSignal, ThresholdRecord]]:
        """Loads a driver's threshold snapshot from disk into the in-memory cache.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            Optional[Dict[TrackedSignal, ThresholdRecord]]: The loaded
            records (also cached), or ``None`` if no snapshot file exists
            or it could not be parsed.
        """
        path = self._file_path(driver_id)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not load threshold snapshot for driver_id=%d from %s: %s",
                driver_id,
                path,
                exc,
            )
            return None

        try:
            records = {
                TrackedSignal(signal_name): ThresholdRecord.from_dict(record_data)
                for signal_name, record_data in payload["thresholds"].items()
            }
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Malformed threshold snapshot for driver_id=%d at %s: %s; ignoring.",
                driver_id,
                path,
                exc,
            )
            return None

        with self._lock:
            self._cache[driver_id] = records

        logger.info(
            "Loaded persisted thresholds for driver_id=%d from %s (%d signals).",
            driver_id,
            path,
            len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Cache / lifecycle management
    # ------------------------------------------------------------------

    def evict(self, driver_id: int) -> bool:
        """Removes a driver's in-memory cached thresholds (disk file untouched).

        Args:
            driver_id: Primary key of the driver profile to evict.

        Returns:
            bool: True if a cached entry was found and removed.
        """
        with self._lock:
            return self._cache.pop(driver_id, None) is not None

    def delete_persisted(self, driver_id: int) -> bool:
        """Deletes a driver's on-disk threshold snapshot and evicts the cache entry.

        Use with care — this permanently discards all learned
        personalisation for the driver's thresholds (the underlying
        embedding/baseline in
        :mod:`~backend.ml.digital_twin.driver_embedding` and
        :mod:`~backend.ml.digital_twin.personalization` is unaffected and
        will simply regenerate fresh thresholds on the next update).

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            bool: True if a persisted file existed and was deleted.
        """
        self.evict(driver_id)
        path = self._file_path(driver_id)
        if path.exists():
            try:
                path.unlink()
                logger.info("Deleted persisted thresholds for driver_id=%d.", driver_id)
                return True
            except OSError as exc:
                logger.error(
                    "Failed to delete threshold snapshot for driver_id=%d: %s",
                    driver_id,
                    exc,
                )
        return False
