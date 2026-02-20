from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pynq import Overlay, allocate


# Defaults for your current repo layout.
DEFAULT_XSA = "dual_cnn.xsa"
DEFAULT_GESTURE_CSV = "augmented_imudata_test.csv"
DEFAULT_VOICE_X = "voice_X_test.npy"
DEFAULT_VOICE_Y = "voice_y_test.npy"
DEFAULT_OUTDIR = "report/evidence_dual"


GESTURE_LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
VOICE_LABELS = ["yes", "no", "go"]

# Gesture scaler constants
GESTURE_MEAN = np.array(
    [-0.79198196, -0.89734741, -1.85433716, -0.04240223, -0.00923926, -0.00578469],
    dtype=np.float32,
)
GESTURE_STD = np.array(
    [55.806826, 103.93487478, 99.69469418, 1.1262504, 1.12243429, 1.03791274],
    dtype=np.float32,
)


@dataclass
class EvalResult:
    total: int
    correct: int
    cm: np.ndarray
    lat_ms: List[float]

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


def q88_pack_u32(x: np.ndarray) -> np.ndarray:
    """Pack float array to signed Q8.8 in low 16 bits of uint32 AXIS words."""
    q = np.round(np.clip(x, -128.0, 127.99609375) * 256.0).astype(np.int32)
    q16 = (q & 0xFFFF).astype(np.uint32)
    return q16


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


def run_dma(dma, in_buf, out_buf, timeout_s: float) -> None:
    in_buf.flush()
    out_buf.invalidate()
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t0 = time.time()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError("DMA timeout")
        time.sleep(0.001)

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()


def stop_core(core) -> None:
    core.write(0x00, 0x00)


def start_core(core) -> None:
    core.write(0x00, 0x01)


def read_gesture_windows(csv_path: str, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mid = int(float(row["measurement_id"]))
                sid = int(float(row["sequence_id"]))
                y = int(float(row["label_id"]))
                feat = np.array(
                    [
                        float(row["gyro_x"]),
                        float(row["gyro_y"]),
                        float(row["gyro_z"]),
                        float(row["acc_x"]),
                        float(row["acc_y"]),
                        float(row["acc_z"]),
                    ],
                    dtype=np.float32,
                )
            except Exception:
                continue
            groups.setdefault(mid, []).append((sid, feat, y))

    windows = []
    labels = []
    for mid in sorted(groups.keys()):
        seq = groups[mid]
        if len(seq) != 60:
            continue
        seq_sorted = sorted(seq, key=lambda x: x[0])
        ids = [s for s, _, _ in seq_sorted]
        if len(set(ids)) != 60:
            continue
        y = seq_sorted[0][2]
        if y < 0 or y >= len(GESTURE_LABELS):
            continue
        x = np.stack([v for _, v, _ in seq_sorted], axis=0)  # [60, 6]
        windows.append(x)
        labels.append(y)
        if max_samples > 0 and len(windows) >= max_samples:
            break

    if not windows:
        raise RuntimeError(f"No valid gesture windows found in {csv_path}")
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def pretty_cm(cm: np.ndarray, labels: Sequence[str]) -> None:
    header = "true\\pred | " + " ".join([f"{l[:6]:>6}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = " ".join([f"{int(cm[i, j]):6d}" for j in range(len(labels))])
        print(f"{label[:9]:>9} | {row}")


def run_gesture_eval(
    gesture_core,
    gesture_dma,
    X: np.ndarray,
    y: np.ndarray,
    timeout_s: float,
    gesture_pack: str,
) -> EvalResult:
    n = len(X)
    cm = np.zeros((len(GESTURE_LABELS), len(GESTURE_LABELS)), dtype=np.int32)
    lat_ms: List[float] = []
    correct = 0

    if gesture_pack == "q88":
        in_buffer = allocate(shape=(60 * 6,), dtype=np.uint32)
    else:
        in_buffer = allocate(shape=(60 * 6,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.uint32)

    try:
        for i in range(n):
            x = (X[i] - GESTURE_MEAN.reshape(1, 6)) / (GESTURE_STD.reshape(1, 6) + 1e-6)
            flat = x.reshape(-1).astype(np.float32)  # [t][c] flatten

            if gesture_pack == "q88":
                payload = q88_pack_u32(flat)
            else:
                payload = flat
            np.copyto(in_buffer, payload)

            t0 = time.time()
            start_core(gesture_core)
            run_dma(gesture_dma, in_buffer, out_buffer, timeout_s=timeout_s)
            lat_ms.append((time.time() - t0) * 1000.0)
            pred = int(out_buffer[0])

            if pred < 0 or pred >= len(GESTURE_LABELS):
                pred = 0
            yt = int(y[i])
            cm[yt, pred] += 1
            correct += int(pred == yt)
    finally:
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass

    return EvalResult(total=n, correct=correct, cm=cm, lat_ms=lat_ms)


def run_voice_eval(
    voice_core,
    voice_dma,
    X: np.ndarray,
    y: np.ndarray,
    timeout_s: float,
    voice_pack: str,
    voice_order: str,
) -> EvalResult:
    n = len(X)
    n_classes = len(VOICE_LABELS)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    lat_ms: List[float] = []
    correct = 0

    if voice_pack == "q88":
        in_buffer = allocate(shape=(40 * 50,), dtype=np.uint32)
    else:
        in_buffer = allocate(shape=(40 * 50,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.uint32)

    try:
        for i in range(n):
            feat = X[i].astype(np.float32)  # [40, 50]
            if voice_order == "tc":
                flat = feat.T.reshape(-1).astype(np.float32)  # [t][c]
            else:
                flat = feat.reshape(-1).astype(np.float32)  # [c][t]

            if voice_pack == "q88":
                payload = q88_pack_u32(flat)
            else:
                payload = flat
            np.copyto(in_buffer, payload)

            t0 = time.time()
            start_core(voice_core)
            run_dma(voice_dma, in_buffer, out_buffer, timeout_s=timeout_s)
            lat_ms.append((time.time() - t0) * 1000.0)
            pred = int(out_buffer[0])

            if pred < 0 or pred >= n_classes:
                pred = 0
            yt = int(y[i])
            if yt < 0 or yt >= n_classes:
                continue
            cm[yt, pred] += 1
            correct += int(pred == yt)
    finally:
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass

    return EvalResult(total=n, correct=correct, cm=cm, lat_ms=lat_ms)


def summarize(result: EvalResult) -> Dict[str, float]:
    lat = np.asarray(result.lat_ms, dtype=np.float64)
    return {
        "total": int(result.total),
        "correct": int(result.correct),
        "accuracy_pct": float(result.accuracy),
        "latency_mean_ms": float(lat.mean()) if lat.size else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
        "latency_p90_ms": float(np.percentile(lat, 90)) if lat.size else 0.0,
        "latency_p99_ms": float(np.percentile(lat, 99)) if lat.size else 0.0,
        "latency_max_ms": float(lat.max()) if lat.size else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Dual CNN hardware test on Ultra96 (PYNQ)")
    p.add_argument("--xsa-path", default=DEFAULT_XSA)

    p.add_argument("--gesture-core", default="gesture_cnn_0")
    p.add_argument("--voice-core", default="voice_cnn_0")
    p.add_argument("--gesture-dma", default="axi_dma_gesture")
    p.add_argument("--voice-dma", default="axi_dma_voice")

    p.add_argument("--gesture-csv", default=DEFAULT_GESTURE_CSV)
    p.add_argument("--voice-features", default=DEFAULT_VOICE_X)
    p.add_argument("--voice-labels", default=DEFAULT_VOICE_Y)
    p.add_argument("--gesture-max-samples", type=int, default=120)
    p.add_argument("--voice-max-samples", type=int, default=300)
    p.add_argument("--timeout-s", type=float, default=2.0)

    p.add_argument("--mode", choices=["gesture", "voice", "both"], default="both")
    p.add_argument("--gesture-pack", choices=["q88", "float32"], default="q88")
    p.add_argument("--voice-pack", choices=["float32", "q88"], default="q88")
    p.add_argument("--voice-order", choices=["tc", "ct"], default="tc")

    p.add_argument("--save-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id = f"{run_id}_{args.tag}"
    out_dir = Path(args.save_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading overlay: {args.xsa_path}")
    overlay = Overlay(args.xsa_path)
    print("Available IP blocks:", ", ".join(sorted(overlay.ip_dict.keys())))

    gesture_core = getattr(overlay, args.gesture_core)
    voice_core = getattr(overlay, args.voice_core)
    gesture_dma = getattr(overlay, args.gesture_dma)
    voice_dma = getattr(overlay, args.voice_dma)

    stop_core(gesture_core)
    stop_core(voice_core)

    reset_dma(gesture_dma)
    if voice_dma is not gesture_dma:
        reset_dma(voice_dma)

    report: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
    }

    try:
        if args.mode in ("gesture", "both"):
            Xg, yg = read_gesture_windows(args.gesture_csv, max_samples=args.gesture_max_samples)
            print(f"\nRunning gesture test: {len(Xg)} samples")
            gres = run_gesture_eval(
                gesture_core=gesture_core,
                gesture_dma=gesture_dma,
                X=Xg,
                y=yg,
                timeout_s=args.timeout_s,
                gesture_pack=args.gesture_pack,
            )
            gsum = summarize(gres)
            print("Gesture accuracy:", f"{gsum['accuracy_pct']:.2f}%")
            pretty_cm(gres.cm, GESTURE_LABELS)
            report["gesture"] = {
                "summary": gsum,
                "confusion_matrix": gres.cm.tolist(),
            }
            stop_core(gesture_core)

        if args.mode in ("voice", "both"):
            Xv = np.load(args.voice_features).astype(np.float32)
            yv = np.load(args.voice_labels).astype(np.int64)
            n = len(Xv)
            if args.voice_max_samples > 0:
                n = min(n, args.voice_max_samples)
            Xv = Xv[:n]
            yv = yv[:n]
            print(f"\nRunning voice test: {len(Xv)} samples")
            vres = run_voice_eval(
                voice_core=voice_core,
                voice_dma=voice_dma,
                X=Xv,
                y=yv,
                timeout_s=args.timeout_s,
                voice_pack=args.voice_pack,
                voice_order=args.voice_order,
            )
            vsum = summarize(vres)
            print("Voice accuracy:", f"{vsum['accuracy_pct']:.2f}%")
            pretty_cm(vres.cm, VOICE_LABELS)
            report["voice"] = {
                "summary": vsum,
                "confusion_matrix": vres.cm.tolist(),
            }
            stop_core(voice_core)

    finally:
        stop_core(gesture_core)
        stop_core(voice_core)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {summary_path}")


if __name__ == "__main__":
    main()
