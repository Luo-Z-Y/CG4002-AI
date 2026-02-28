"""
router.py

PS-side router for dual-IP runtime arbitration (gesture + voice) on Ultra96.

Design goals:
- Differentiate gesture vs voice in real time using lightweight scores.
- Start only the selected IP core (avoid auto-starting every HLS core).
- Support both separate DMAs and shared DMA setup.
- Keep existing gesture/voice scripts untouched.

Notes:
- This is a routing scaffold. Replace fetch_imu_window()/fetch_audio_chunk()
  with your real sensors or IPC queues.
- Gesture input expected shape: [60, 6] (time-major, normalized as needed).
- Voice input expected shape: [40, 50] MFCC. The router reorders to match
  voice HLS stream order (t-major, c-inner) before DMA.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pynq import Overlay, allocate


@dataclass
class RouterConfig:
    xsa_path: str
    gesture_core: str
    voice_core: str
    gesture_dma: str
    voice_dma: str
    timeout_s: float
    motion_thr: float
    voice_thr: float
    priority: str
    cooldown_ms: float
    loop_hz: float
    demo_mode: bool


class DualIPRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.overlay = Overlay(cfg.xsa_path)

        self.gesture_core = getattr(self.overlay, cfg.gesture_core)
        self.voice_core = getattr(self.overlay, cfg.voice_core)

        self.gesture_dma = getattr(self.overlay, cfg.gesture_dma)
        self.voice_dma = getattr(self.overlay, cfg.voice_dma)

        # Input buffer sizes are fixed by your HLS interfaces.
        self.gesture_in = allocate(shape=(60 * 6,), dtype=np.float32)
        self.gesture_out = allocate(shape=(1,), dtype=np.int32)

        self.voice_in = allocate(shape=(40 * 50,), dtype=np.float32)
        self.voice_out = allocate(shape=(1,), dtype=np.int32)

        self.last_route_ts = 0.0

        self.reset_dma(self.gesture_dma)
        if self.voice_dma is not self.gesture_dma:
            self.reset_dma(self.voice_dma)

        # Ensure both cores are stopped at init.
        self.stop_core(self.gesture_core)
        self.stop_core(self.voice_core)

    def close(self) -> None:
        try:
            self.gesture_in.close()
            self.gesture_out.close()
            self.voice_in.close()
            self.voice_out.close()
        except Exception:
            pass

    @staticmethod
    def reset_dma(dma) -> None:
        for ch in [dma.sendchannel, dma.recvchannel]:
            mmio = ch._mmio
            off = ch._offset
            mmio.write(off + 0x00, 0x4)  # reset
        time.sleep(0.01)
        for ch in [dma.sendchannel, dma.recvchannel]:
            mmio = ch._mmio
            off = ch._offset
            mmio.write(off + 0x00, 0x1)  # run
        time.sleep(0.01)

    @staticmethod
    def start_core(core) -> None:
        # ap_start=1, auto_restart=0 for single-shot dispatch
        core.write(0x00, 0x01)

    @staticmethod
    def stop_core(core) -> None:
        # clear ap_start/auto_restart
        core.write(0x00, 0x00)

    @staticmethod
    def run_dma_with_timeout(dma, in_buf, out_buf, timeout_s: float) -> None:
        in_buf.flush()
        out_buf.invalidate()

        dma.recvchannel.transfer(out_buf)
        dma.sendchannel.transfer(in_buf)

        t0 = time.time()
        while True:
            if dma.sendchannel.idle and dma.recvchannel.idle:
                break
            if time.time() - t0 > timeout_s:
                raise TimeoutError("DMA stalled")
            time.sleep(0.001)

        dma.sendchannel.wait()
        dma.recvchannel.wait()
        out_buf.invalidate()

    @staticmethod
    def motion_score(imu_window_60x6: np.ndarray) -> float:
        # Lightweight score used only for routing decision.
        # IMU columns: [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        gyro = imu_window_60x6[:, :3]
        acc = imu_window_60x6[:, 3:6]
        g = np.sqrt(np.sum(gyro * gyro, axis=1)).mean()
        a = np.sqrt(np.sum(acc * acc, axis=1)).mean()
        return float(g + 0.5 * a)

    @staticmethod
    def voice_score(audio_chunk: np.ndarray) -> float:
        # RMS energy for VAD-like gate.
        x = audio_chunk.reshape(-1).astype(np.float32)
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def infer_gesture(self, imu_window_60x6: np.ndarray) -> int:
        # Required shape: [60, 6], flattened in time-major order.
        flat = imu_window_60x6.astype(np.float32).reshape(-1)
        np.copyto(self.gesture_in, flat)

        self.stop_core(self.voice_core)
        self.start_core(self.gesture_core)
        self.run_dma_with_timeout(
            self.gesture_dma,
            self.gesture_in,
            self.gesture_out,
            self.cfg.timeout_s,
        )
        return int(self.gesture_out[0])

    def infer_voice(self, mfcc_40x50: np.ndarray) -> int:
        # Voice HLS input loop is [t][c], so reorder from [c, t] -> [t, c] before flatten.
        x_tc = mfcc_40x50.astype(np.float32).T  # [50, 40]
        flat = x_tc.reshape(-1)
        np.copyto(self.voice_in, flat)

        self.stop_core(self.gesture_core)
        self.start_core(self.voice_core)
        self.run_dma_with_timeout(
            self.voice_dma,
            self.voice_in,
            self.voice_out,
            self.cfg.timeout_s,
        )
        return int(self.voice_out[0])

    def arbitrate(self, m_score: float, v_score: float) -> str:
        # cooldown to prevent rapid mode flapping
        now = time.time() * 1000.0
        if now - self.last_route_ts < self.cfg.cooldown_ms:
            return "none"

        m_on = m_score > self.cfg.motion_thr
        v_on = v_score > self.cfg.voice_thr

        if m_on and not v_on:
            return "gesture"
        if v_on and not m_on:
            return "voice"
        if m_on and v_on:
            return self.cfg.priority
        return "none"

    def mark_dispatched(self) -> None:
        self.last_route_ts = time.time() * 1000.0


# ------------------------------------------------------------------
# Data source hooks (replace these with real sensor integration)
# ------------------------------------------------------------------
def fetch_imu_window() -> Optional[np.ndarray]:
    # Return np.float32 [60, 6] when available; otherwise None.
    return None


def fetch_audio_chunk() -> Optional[np.ndarray]:
    # Return recent mono waveform chunk np.float32 [N] when available; otherwise None.
    return None


def demo_inputs(step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Synthetic demo feed: every few steps trigger gesture or voice.
    imu = np.random.normal(0.0, 0.02, size=(60, 6)).astype(np.float32)
    audio = np.random.normal(0.0, 0.002, size=(16000,)).astype(np.float32)

    if step % 30 in (5, 6, 7):
        imu += 2.0  # strong motion segment
    if step % 30 in (18, 19, 20):
        audio += 0.08 * np.sin(np.linspace(0, 400 * np.pi, audio.shape[0])).astype(np.float32)

    # For demo, fake MFCC [40,50]. In real use, get from your voice preprocessor.
    mfcc = np.random.normal(0.0, 1.0, size=(40, 50)).astype(np.float32)
    return imu, audio, mfcc


def parse_args() -> RouterConfig:
    p = argparse.ArgumentParser(description="Dual-IP real-time router (gesture vs voice)")
    p.add_argument("--xsa-path", required=True, help="Overlay path containing both gesture and voice IP")
    p.add_argument("--gesture-core", default="gesture_cnn_0", help="Overlay IP name for gesture core")
    p.add_argument("--voice-core", default="voice_cnn_0", help="Overlay IP name for voice core")
    p.add_argument("--gesture-dma", default="axi_dma_gesture", help="DMA name for gesture stream")
    p.add_argument("--voice-dma", default="axi_dma_voice", help="DMA name for voice stream")
    p.add_argument("--timeout-s", type=float, default=2.0)
    p.add_argument("--motion-thr", type=float, default=1.2, help="Routing threshold for motion score")
    p.add_argument("--voice-thr", type=float, default=0.015, help="Routing threshold for voice RMS")
    p.add_argument("--priority", choices=["gesture", "voice"], default="gesture")
    p.add_argument("--cooldown-ms", type=float, default=150.0, help="Min dispatch interval")
    p.add_argument("--loop-hz", type=float, default=50.0)
    p.add_argument("--demo-mode", action="store_true", help="Use synthetic inputs for routing sanity test")
    args = p.parse_args()

    return RouterConfig(
        xsa_path=args.xsa_path,
        gesture_core=args.gesture_core,
        voice_core=args.voice_core,
        gesture_dma=args.gesture_dma,
        voice_dma=args.voice_dma,
        timeout_s=args.timeout_s,
        motion_thr=args.motion_thr,
        voice_thr=args.voice_thr,
        priority=args.priority,
        cooldown_ms=args.cooldown_ms,
        loop_hz=args.loop_hz,
        demo_mode=args.demo_mode,
    )


def main() -> None:
    cfg = parse_args()
    router = DualIPRouter(cfg)

    print("Router started.")
    print(f"Motion threshold={cfg.motion_thr}, Voice threshold={cfg.voice_thr}, Priority={cfg.priority}")

    step = 0
    period_s = 1.0 / max(cfg.loop_hz, 1e-6)

    try:
        while True:
            t0 = time.time()

            if cfg.demo_mode:
                imu_win, audio_chunk, mfcc = demo_inputs(step)
            else:
                imu_win = fetch_imu_window()
                audio_chunk = fetch_audio_chunk()
                mfcc = None  # set this via your own voice preprocessor pipeline

            m_score = router.motion_score(imu_win) if imu_win is not None else 0.0
            v_score = router.voice_score(audio_chunk) if audio_chunk is not None else 0.0

            decision = router.arbitrate(m_score, v_score)

            if decision == "gesture" and imu_win is not None:
                pred = router.infer_gesture(imu_win)
                router.mark_dispatched()
                print(f"[ROUTE] gesture | m={m_score:.3f} v={v_score:.3f} -> pred={pred}")

            elif decision == "voice" and mfcc is not None:
                pred = router.infer_voice(mfcc)
                router.mark_dispatched()
                print(f"[ROUTE] voice   | m={m_score:.3f} v={v_score:.3f} -> pred={pred}")

            else:
                if step % 20 == 0:
                    print(f"[IDLE] m={m_score:.3f} v={v_score:.3f}")

            step += 1
            dt = time.time() - t0
            if dt < period_s:
                time.sleep(period_s - dt)

    except KeyboardInterrupt:
        print("Router stopped.")
    finally:
        router.close()


if __name__ == "__main__":
    main()
