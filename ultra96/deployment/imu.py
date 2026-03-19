from __future__ import annotations

"""IMU preprocessing helpers for Ultra96 deployment."""

import numpy as np

from messages import ImuData, ImuSample, build_imu_data


class ImuPreprocessor:
    """Validate variable-length IMU windows and resample them to the model length."""

    def __init__(self, target_count: int = 60, min_count: int = 15, max_count: int = 60) -> None:
        if min_count <= 0:
            raise ValueError("min_count must be positive")
        if max_count < min_count:
            raise ValueError("max_count must be >= min_count")
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        self.target_count = int(target_count)
        self.min_count = int(min_count)
        self.max_count = int(max_count)

    def preprocess(self, samples) -> ImuData:
        data = samples if isinstance(samples, ImuData) else build_imu_data(samples)
        count = data.count if data.count is not None else len(data.samples)
        if not self.min_count <= count <= self.max_count:
            raise ValueError(
                f"IMU payload must contain between {self.min_count} and {self.max_count} samples, got {count}"
            )

        if count == self.target_count:
            return data

        window = np.asarray(
            [[sample.y, sample.p, sample.r, sample.ax, sample.ay, sample.az] for sample in data.samples],
            dtype=np.float32,
        )
        resampled = self._resample_window(window)
        return ImuData(
            samples=[
                ImuSample(
                    y=float(row[0]),
                    p=float(row[1]),
                    r=float(row[2]),
                    ax=float(row[3]),
                    ay=float(row[4]),
                    az=float(row[5]),
                )
                for row in resampled
            ],
            count=self.target_count,
        )

    def _resample_window(self, window: np.ndarray) -> np.ndarray:
        source_count = int(window.shape[0])
        if source_count == self.target_count:
            return window.astype(np.float32)

        t_old = np.linspace(0.0, 1.0, source_count, dtype=np.float32)
        t_new = np.linspace(0.0, 1.0, self.target_count, dtype=np.float32)
        out = np.empty((self.target_count, 6), dtype=np.float32)
        for col_idx in range(6):
            out[:, col_idx] = np.interp(t_new, t_old, window[:, col_idx]).astype(np.float32)
        return out
