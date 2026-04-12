from __future__ import annotations

"""Single-sample hardware inference wrapper for the EC2 bridge."""

from typing import Optional, Sequence

import numpy as np

try:
    from . import hardware as hw
    from .messages import ClassificationData, ImuData, VoiceMfccData, build_imu_data, build_voice_mfcc_data
    from .imu import load_feature_norm_stats, normalize_window
except ImportError:
    import hardware as hw
    from messages import ClassificationData, ImuData, VoiceMfccData, build_imu_data, build_voice_mfcc_data
    from imu import load_feature_norm_stats, normalize_window

class Ultra96Runtime:
    """Wrap the deployed overlay and expose one-request-at-a-time inference calls.

    The current overlay returns only the winning class index. The bridge therefore
    emits a configurable placeholder confidence until score output exists in PL.
    """

    def __init__(
        self,
        xsa_path: str = hw.DEFAULT_XSA,
        gesture_core_name: str = hw.GESTURE_CORE_NAME,
        voice_core_name: str = hw.VOICE_CORE_NAME,
        gesture_dma_name: str = hw.GESTURE_DMA_NAME,
        voice_dma_name: str = hw.VOICE_DMA_NAME,
        timeout_s: float = 2.0,
        gesture_mean: str | None = None,
        gesture_std: str | None = None,
        gesture_labels: Optional[Sequence[str]] = None,
        voice_labels: Optional[Sequence[str]] = None,
        default_confidence: float = 1.0,
    ) -> None:
        from pynq import Overlay

        self.timeout_s = timeout_s
        self.gesture_labels = list(gesture_labels or hw.GESTURE_LABELS)
        self.voice_labels = list(voice_labels or hw.VOICE_LABELS)
        self.default_confidence = float(default_confidence)
        self.gesture_mean = None
        self.gesture_std = None
        if (gesture_mean is None) != (gesture_std is None):
            raise ValueError("gesture_mean and gesture_std must be provided together")
        if gesture_mean is not None and gesture_std is not None:
            self.gesture_mean, self.gesture_std = load_feature_norm_stats(gesture_mean, gesture_std)

        self.overlay = Overlay(xsa_path)
        self.gesture_core = getattr(self.overlay, gesture_core_name)
        self.voice_core = getattr(self.overlay, voice_core_name)
        self.gesture_dma = getattr(self.overlay, gesture_dma_name)
        self.voice_dma = getattr(self.overlay, voice_dma_name)

        self.reset()

    def reset(self) -> None:
        hw.stop_core(self.gesture_core)
        hw.stop_core(self.voice_core)
        hw.reset_dma(self.gesture_dma)
        if self.voice_dma is not self.gesture_dma:
            hw.reset_dma(self.voice_dma)

    def close(self) -> None:
        hw.stop_core(self.gesture_core)
        hw.stop_core(self.voice_core)

    def classify_imu(self, samples) -> ClassificationData:
        data = samples if isinstance(samples, ImuData) else build_imu_data(samples)
        if data.count != 60:
            raise ValueError(f"Ultra96 gesture runtime expects exactly 60 IMU samples, got {data.count}")

        window = np.asarray(
            [[sample.gx, sample.gy, sample.gz, sample.ax, sample.ay, sample.az] for sample in data.samples],
            dtype=np.float32,
        )
        pred_idx = self._predict_gesture(window)
        return ClassificationData(label=self._resolve_label(pred_idx, self.gesture_labels), confidence=self.default_confidence)

    def classify_voice_mfcc(self, features) -> ClassificationData:
        data = features if isinstance(features, VoiceMfccData) else build_voice_mfcc_data(features)
        matrix = np.asarray(data.features, dtype=np.float32)
        pred_idx = self._predict_voice(matrix)
        return ClassificationData(label=self._resolve_label(pred_idx, self.voice_labels), confidence=self.default_confidence)

    def _predict_gesture(self, window: np.ndarray) -> int:
        if self.gesture_mean is not None and self.gesture_std is not None:
            window = normalize_window(window, self.gesture_mean, self.gesture_std)
        flat = window.reshape(-1).astype(np.float32)
        return self._run_core(self.gesture_core, self.gesture_dma, flat)

    def _predict_voice(self, mfcc: np.ndarray) -> int:
        flat = mfcc.T.reshape(-1).astype(np.float32)
        return self._run_core(self.voice_core, self.voice_dma, flat)

    def _run_core(self, core, dma, flat: np.ndarray) -> int:
        from pynq import allocate

        in_buffer = allocate(shape=(flat.size,), dtype=np.uint32)
        out_buffer = allocate(shape=(1,), dtype=np.uint32)
        try:
            np.copyto(in_buffer, hw.q88_pack_u32(flat))
            hw.start_core(core)
            hw.run_dma(dma, in_buffer, out_buffer, timeout_s=self.timeout_s)
            return int(out_buffer[0])
        except Exception:
            # A timed-out transaction can leave the core/DMA path wedged. Reset the
            # active path so later requests can still make progress.
            try:
                hw.stop_core(core)
            except Exception:
                pass
            try:
                hw.reset_dma(dma)
            except Exception:
                pass
            raise
        finally:
            try:
                hw.stop_core(core)
            except Exception:
                pass
            try:
                in_buffer.close()
                out_buffer.close()
            except Exception:
                pass

    @staticmethod
    def _resolve_label(pred_idx: int, labels: Sequence[str]) -> str:
        if 0 <= pred_idx < len(labels):
            return labels[pred_idx]
        return labels[0]
