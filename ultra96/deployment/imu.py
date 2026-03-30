from __future__ import annotations

"""IMU preprocessing helpers for Ultra96 deployment.

Pipeline (matches notebook train_gesture_cnn.ipynb):
1. Validate sample count against [min_count, max_count].
2. Motion-based trim: remove leading/trailing idle frames using gyro magnitude threshold.
3. Reject if trimmed length falls outside [trim_min, trim_max].
4. Per-window baseline removal: subtract mean of first ``baseline_frames`` per channel.
5. FFT-based resample to ``target_count`` (same implementation used in training).
6. Return preprocessed ImuData ready for normalisation and inference.
"""

import numpy as np

from messages import ImuData, ImuSample, build_imu_data

# Feature order — must match training CSV column order.
FEATURE_ORDER = ("gx", "gy", "gz", "ax", "ay", "az")


class ImuPreprocessor:
    """Validate variable-length IMU windows, trim, baseline-remove, and resample."""

    def __init__(
        self,
        target_count: int = 60,
        min_count: int = 15,
        max_count: int = 300,
        trim_min: int = 40,
        trim_max: int = 300,
        gyro_idle_threshold: float = 5.0,
        baseline_frames: int = 5,
    ) -> None:
        if min_count <= 0:
            raise ValueError("min_count must be positive")
        if max_count < min_count:
            raise ValueError("max_count must be >= min_count")
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        self.target_count = int(target_count)
        self.min_count = int(min_count)
        self.max_count = int(max_count)
        self.trim_min = int(trim_min)
        self.trim_max = int(trim_max)
        self.gyro_idle_threshold = float(gyro_idle_threshold)
        self.baseline_frames = int(baseline_frames)
        self.last_debug = None
        self.last_capture = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, samples) -> ImuData:
        data = samples if isinstance(samples, ImuData) else build_imu_data(samples)
        count = data.count if data.count is not None else len(data.samples)
        if not self.min_count <= count <= self.max_count:
            raise ValueError(
                f"IMU payload must contain between {self.min_count} and {self.max_count} samples, got {count}"
            )

        window = np.asarray(
            [[getattr(s, f) for f in FEATURE_ORDER] for s in data.samples],
            dtype=np.float32,
        )

        raw_window = window.copy()

        # 1. Motion-based trim
        window = self._trim_idle(window)
        trimmed_count = int(window.shape[0])

        if not self.trim_min <= trimmed_count <= self.trim_max:
            raise ValueError(
                f"Trimmed window length {trimmed_count} outside valid band "
                f"[{self.trim_min}, {self.trim_max}]"
            )

        # 2. Per-window baseline removal
        window = self._remove_baseline(window)

        # 3. Resample to target length
        if trimmed_count != self.target_count:
            resampled = self._resample_window(window)
        else:
            resampled = window.astype(np.float32)

        self.last_debug = {
            "raw_count": int(count),
            "trimmed_count": trimmed_count,
            "out_count": int(self.target_count),
            "raw": self._stats(raw_window),
            "out": self._stats(resampled),
            "raw_first": [float(v) for v in raw_window[0]],
            "raw_last": [float(v) for v in raw_window[-1]],
            "out_first": [float(v) for v in resampled[0]],
            "out_last": [float(v) for v in resampled[-1]],
        }
        self.last_capture = {
            "raw_window": raw_window,
            "resampled_window": resampled.copy(),
        }

        return ImuData(
            samples=[
                ImuSample(
                    gx=float(row[0]),
                    gy=float(row[1]),
                    gz=float(row[2]),
                    ax=float(row[3]),
                    ay=float(row[4]),
                    az=float(row[5]),
                )
                for row in resampled
            ],
            count=self.target_count,
        )

    # ------------------------------------------------------------------
    # Trim
    # ------------------------------------------------------------------

    def _trim_idle(self, window: np.ndarray) -> np.ndarray:
        """Remove leading/trailing frames where gyro magnitude is below threshold."""
        gyro = window[:, :3]  # gx, gy, gz
        mag = np.sqrt((gyro ** 2).sum(axis=1))
        active = mag >= self.gyro_idle_threshold

        if not active.any():
            # No motion detected — return the full window and let the caller decide.
            return window

        indices = np.where(active)[0]
        start = int(indices[0])
        end = int(indices[-1]) + 1
        return window[start:end]

    # ------------------------------------------------------------------
    # Baseline removal
    # ------------------------------------------------------------------

    def _remove_baseline(self, window: np.ndarray) -> np.ndarray:
        """Subtract mean of the first N frames per channel (especially helpful for accel)."""
        n = min(self.baseline_frames, window.shape[0])
        baseline = window[:n].mean(axis=0, keepdims=True)
        return window - baseline

    # ------------------------------------------------------------------
    # FFT resample (matches scipy.signal.resample behaviour)
    # ------------------------------------------------------------------

    @staticmethod
    def _fft_resample_1d(signal_1d: np.ndarray, target_count: int) -> np.ndarray:
        """Fourier-domain resample aligned with scipy.signal.resample policy."""
        source_count = int(signal_1d.shape[0])
        if source_count == target_count:
            return signal_1d.astype(np.float32)

        spectrum = np.fft.fft(signal_1d.astype(np.float64), axis=0)
        resized = np.zeros((target_count,), dtype=np.complex128)

        keep = min(source_count, target_count)
        half = keep // 2
        if keep % 2 == 0:
            resized[:half] = spectrum[:half]
            resized[-half:] = spectrum[-half:]
            if half < spectrum.shape[0] and half < resized.shape[0]:
                resized[half] = spectrum[half]
        else:
            resized[: half + 1] = spectrum[: half + 1]
            resized[-half:] = spectrum[-half:]

        resampled = np.fft.ifft(resized, axis=0).real
        resampled *= float(target_count) / float(source_count)
        return resampled.astype(np.float32)

    def _resample_window(self, window: np.ndarray) -> np.ndarray:
        source_count = int(window.shape[0])
        if source_count == self.target_count:
            return window.astype(np.float32)

        out = np.empty((self.target_count, 6), dtype=np.float32)
        for col_idx in range(6):
            out[:, col_idx] = self._fft_resample_1d(window[:, col_idx], self.target_count)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stats(window: np.ndarray) -> dict[str, object]:
        return {
            "shape": tuple(int(dim) for dim in window.shape),
            "min": float(window.min()),
            "max": float(window.max()),
            "mean": float(window.mean()),
            "std": float(window.std()),
        }
