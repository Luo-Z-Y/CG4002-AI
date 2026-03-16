from __future__ import annotations

"""Audio preprocessing helpers for the EC2 side.

The chosen architecture is:
- Viz sends `.m4a` audio to EC2
- EC2 decodes `.m4a` to mono PCM
- EC2 extracts MFCC features with shape [40, 50]
- EC2 sends `voice_mfcc` to Ultra96

This file intentionally keeps MFCC extraction in NumPy so it does not depend on
heavy audio Python packages.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np


class VoicePreprocessor:
    """Numpy-based MFCC extractor matching the current voice model input contract."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        n_mfcc: int = 40,
        target_frames: int = 50,
        pre_emphasis: float = 0.97,
        eps: float = 1e-8,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_frames = target_frames
        self.pre_emphasis = pre_emphasis
        self.eps = eps

        self.window = np.hanning(self.win_length).astype(np.float32)
        self.mel_fb = self._build_mel_filterbank()
        self.dct_mat = self._build_dct_matrix()

    @staticmethod
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        # Build a fixed mel filterbank once so repeated calls only do the signal work.
        n_freqs = self.n_fft // 2 + 1
        f_max = float(self.sample_rate) / 2.0

        mel_min = self._hz_to_mel(np.array([0.0], dtype=np.float32))[0]
        mel_max = self._hz_to_mel(np.array([f_max], dtype=np.float32))[0]
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2, dtype=np.float32)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(np.int32)

        fb = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for mel_idx in range(1, self.n_mels + 1):
            left = bins[mel_idx - 1]
            center = bins[mel_idx]
            right = bins[mel_idx + 1]

            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1

            for freq_idx in range(left, center):
                if 0 <= freq_idx < n_freqs:
                    fb[mel_idx - 1, freq_idx] = (freq_idx - left) / float(center - left)
            for freq_idx in range(center, right):
                if 0 <= freq_idx < n_freqs:
                    fb[mel_idx - 1, freq_idx] = (right - freq_idx) / float(right - center)

        norm = 2.0 / np.maximum(hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels], self.eps)
        fb *= norm[:, None]
        return fb

    def _build_dct_matrix(self) -> np.ndarray:
        # DCT-II projects log-mel features into MFCC coefficients.
        n = np.arange(self.n_mels, dtype=np.float32)
        k = np.arange(self.n_mfcc, dtype=np.float32)[:, None]
        dct = np.cos(np.pi / self.n_mels * (n + 0.5) * k).astype(np.float32)
        dct[0] *= 1.0 / np.sqrt(2.0)
        dct *= np.sqrt(2.0 / self.n_mels)
        return dct

    def _pre_emphasize(self, waveform: np.ndarray) -> np.ndarray:
        # Standard voice preprocessing step to emphasize higher frequencies slightly.
        if waveform.size == 0:
            return waveform.astype(np.float32)
        emphasized = np.empty_like(waveform, dtype=np.float32)
        emphasized[0] = waveform[0]
        emphasized[1:] = waveform[1:] - self.pre_emphasis * waveform[:-1]
        return emphasized

    def _frame(self, waveform: np.ndarray) -> np.ndarray:
        # Slice the waveform into overlapping windows for STFT processing.
        if waveform.size < self.win_length:
            waveform = np.pad(waveform, (0, self.win_length - waveform.size), mode="constant")

        frame_count = 1 + int(np.floor((waveform.size - self.win_length) / self.hop_length))
        frame_count = max(frame_count, 1)
        total_len = (frame_count - 1) * self.hop_length + self.win_length
        if waveform.size < total_len:
            waveform = np.pad(waveform, (0, total_len - waveform.size), mode="constant")

        shape = (frame_count, self.win_length)
        strides = (waveform.strides[0] * self.hop_length, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        return frames.copy()

    def _stft_power(self, waveform: np.ndarray) -> np.ndarray:
        # Produce a power spectrogram with shape [freq_bins, frames].
        frames = self._frame(waveform) * self.window[None, :]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)
        return (np.abs(spec) ** 2).astype(np.float32).T

    def _pad_or_trim_time(self, feat: np.ndarray) -> np.ndarray:
        # The Ultra96 voice model expects a fixed number of time frames: 50.
        frame_count = feat.shape[1]
        if frame_count == self.target_frames:
            return feat
        if frame_count > self.target_frames:
            return feat[:, : self.target_frames]
        return np.pad(feat, ((0, 0), (0, self.target_frames - frame_count)), mode="constant")

    def _cmvn(self, feat: np.ndarray) -> np.ndarray:
        # Normalize each MFCC coefficient across time.
        mean = feat.mean(axis=1, keepdims=True)
        std = feat.std(axis=1, keepdims=True)
        return (feat - mean) / (std + self.eps)

    def process_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert a mono waveform into the fixed MFCC matrix sent to Ultra96."""

        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected sample_rate={self.sample_rate}, got {sample_rate}")

        # MFCC pipeline: pre-emphasis -> STFT power -> mel -> log -> DCT -> fixed frame count -> CMVN
        waveform = self._pre_emphasize(waveform)
        power_spec = self._stft_power(waveform)
        mel_spec = np.matmul(self.mel_fb, power_spec)
        log_mel = np.log(mel_spec + self.eps)
        mfcc = np.matmul(self.dct_mat, log_mel).astype(np.float32)
        mfcc = self._pad_or_trim_time(mfcc)
        return self._cmvn(mfcc).astype(np.float32)


def decode_m4a_to_waveform(
    m4a_path: Union[str, Path],
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> np.ndarray:
    """Decode `.m4a` into mono float32 PCM using the `ffmpeg` executable."""

    if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
        raise FileNotFoundError(
            f"ffmpeg executable not found: {ffmpeg_path}. Install ffmpeg or pass its full path."
        )

    source_path = Path(m4a_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(source_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Decoded empty waveform from {source_path}")
    return waveform


def m4a_to_mfcc_matrix(
    m4a_path: Union[str, Path],
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    preprocessor: Optional[VoicePreprocessor] = None,
) -> np.ndarray:
    """One-shot helper for the full EC2 voice preprocessing path."""

    waveform = decode_m4a_to_waveform(m4a_path, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path)
    processor = preprocessor or VoicePreprocessor(sample_rate=sample_rate)
    return processor.process_waveform(waveform, sample_rate)
