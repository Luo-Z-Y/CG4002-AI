"""
voice_preprocess.py

Standalone voice feature processor for the CG4002 project.
It does not modify or depend on the existing gesture pipeline.

Output format is aligned with the legacy VoiceCNN input:
  [n_mfcc=40, target_frames=50]
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


class VoicePreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,  # 25 ms @ 16 kHz
        hop_length: int = 160,  # 10 ms @ 16 kHz
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
        n_freqs = self.n_fft // 2 + 1
        f_min = 0.0
        f_max = float(self.sample_rate) / 2.0

        mel_min = self._hz_to_mel(np.array([f_min], dtype=np.float32))[0]
        mel_max = self._hz_to_mel(np.array([f_max], dtype=np.float32))[0]
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2, dtype=np.float32)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(np.int32)

        fb = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            left = bins[m - 1]
            center = bins[m]
            right = bins[m + 1]

            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1

            for k in range(left, center):
                if 0 <= k < n_freqs:
                    fb[m - 1, k] = (k - left) / float(center - left)
            for k in range(center, right):
                if 0 <= k < n_freqs:
                    fb[m - 1, k] = (right - k) / float(right - center)

        # Slaney-style area normalization.
        enorm = 2.0 / np.maximum(hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels], self.eps)
        fb *= enorm[:, None]
        return fb

    def _build_dct_matrix(self) -> np.ndarray:
        # DCT-II matrix: [n_mfcc, n_mels]
        n = np.arange(self.n_mels, dtype=np.float32)
        k = np.arange(self.n_mfcc, dtype=np.float32)[:, None]
        dct = np.cos(np.pi / self.n_mels * (n + 0.5) * k).astype(np.float32)
        dct[0] *= 1.0 / np.sqrt(2.0)
        dct *= np.sqrt(2.0 / self.n_mels)
        return dct

    def _pre_emphasize(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        y = np.empty_like(x, dtype=np.float32)
        y[0] = x[0]
        y[1:] = x[1:] - self.pre_emphasis * x[:-1]
        return y

    def _frame(self, x: np.ndarray) -> np.ndarray:
        if x.size < self.win_length:
            pad = self.win_length - x.size
            x = np.pad(x, (0, pad), mode="constant")

        n_frames = 1 + int(np.floor((x.size - self.win_length) / self.hop_length))
        if n_frames <= 0:
            n_frames = 1

        total_len = (n_frames - 1) * self.hop_length + self.win_length
        if x.size < total_len:
            x = np.pad(x, (0, total_len - x.size), mode="constant")

        shape = (n_frames, self.win_length)
        strides = (x.strides[0] * self.hop_length, x.strides[0])
        frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return frames.copy()

    def _stft_power(self, x: np.ndarray) -> np.ndarray:
        frames = self._frame(x)
        frames = frames * self.window[None, :]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power = (np.abs(spec) ** 2).astype(np.float32)
        return power.T  # [freq_bins, frames]

    def _pad_or_trim_time(self, feat: np.ndarray) -> np.ndarray:
        # feat shape: [n_features, time]
        t = feat.shape[1]
        if t == self.target_frames:
            return feat
        if t > self.target_frames:
            return feat[:, : self.target_frames]
        pad_width = self.target_frames - t
        return np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")

    def _cmvn(self, feat: np.ndarray) -> np.ndarray:
        # Per-coefficient normalization across time.
        mean = feat.mean(axis=1, keepdims=True)
        std = feat.std(axis=1, keepdims=True)
        return (feat - mean) / (std + self.eps)

    def process_waveform(self, waveform: np.ndarray, sr: int) -> dict:
        """
        Args:
            waveform: mono float waveform in range roughly [-1, 1]
            sr: sample rate

        Returns:
            {
              "mfcc_40x50": np.float32 [40, 50],
              "flat2000": np.float32 [2000],
              "meta": {...}
            }
        """
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)

        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: got {sr}, expected {self.sample_rate}. "
                "Resample before processing."
            )

        x = self._pre_emphasize(x)
        power_spec = self._stft_power(x)  # [freq, time]

        mel_spec = np.matmul(self.mel_fb, power_spec)  # [n_mels, time]
        log_mel = np.log(mel_spec + self.eps)
        mfcc = np.matmul(self.dct_mat, log_mel).astype(np.float32)  # [n_mfcc, time]

        mfcc = self._pad_or_trim_time(mfcc)
        mfcc = self._cmvn(mfcc).astype(np.float32)

        return {
            "mfcc_40x50": mfcc,
            "flat2000": mfcc.reshape(-1).astype(np.float32),
            "meta": {
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "n_mfcc": self.n_mfcc,
                "target_frames": self.target_frames,
            },
        }

    def process_wav_file(self, wav_path: str | Path) -> dict:
        wav_path = Path(wav_path)
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            fr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if channels != 1:
            raise ValueError(f"Expected mono WAV. Got channels={channels}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV. Got sample width={sampwidth} bytes")

        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return self.process_waveform(x, fr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed-size MFCC features [40, 50] from WAV.")
    parser.add_argument("wav", help="Path to mono 16-bit PCM WAV file")
    parser.add_argument("--out-npy", default="voice_mfcc_40x50.npy", help="Output .npy path")
    parser.add_argument("--sr", type=int, default=16000, help="Expected sample rate")
    args = parser.parse_args()

    pre = VoicePreprocessor(sample_rate=args.sr)
    out = pre.process_wav_file(args.wav)
    np.save(args.out_npy, out["mfcc_40x50"])

    print("✅ Voice preprocessing complete")
    print(f"Input WAV: {args.wav}")
    print(f"Saved: {args.out_npy}")
    print(f"MFCC shape: {out['mfcc_40x50'].shape}")


if __name__ == "__main__":
    main()
