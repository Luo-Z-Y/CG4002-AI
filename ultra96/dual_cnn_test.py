from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pynq import Overlay, allocate


# Defaults for your current repo layout.
DEFAULT_XSA = "dual_cnn.xsa"
DEFAULT_GESTURE_X = "gesture_X_test.npy"
DEFAULT_GESTURE_Y = "gesture_y_test.npy"
DEFAULT_VOICE_X = "voice_X_test.npy"
DEFAULT_VOICE_Y = "voice_y_test.npy"
DEFAULT_OUTDIR = "report/evidence_dual"


DEFAULT_GESTURE_LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
VOICE_LABELS = ["yes", "no", "go"]


@dataclass
class EvalResult:
    # Number of evaluated samples.
    total: int
    # Number of correct predictions.
    correct: int
    # Confusion matrix [true, pred].
    cm: np.ndarray
    # Per-sample end-to-end latency in ms (start core + DMA send/recv).
    lat_ms: List[float]

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


def q88_pack_u32(x: np.ndarray) -> np.ndarray:
    """Pack float array to signed Q8.8 in low 16 bits of uint32 AXIS words."""
    # Clip to representable Q8.8 range, then scale by 2^8.
    q = np.round(np.clip(x, -128.0, 127.99609375) * 256.0).astype(np.int32)
    # Keep only low 16 bits (two's complement int16 payload in AXIS data[15:0]).
    q16 = (q & 0xFFFF).astype(np.uint32)
    return q16


def reset_dma(dma) -> None:
    # Reset both TX (MM2S) and RX (S2MM) channels.
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x4)  # reset
    time.sleep(0.01)
    # Put both channels back into run state.
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x1)  # run
    time.sleep(0.01)


def run_dma(dma, in_buf, out_buf, timeout_s: float) -> None:
    # Ensure input is committed to memory and output cache is invalidated.
    in_buf.flush()
    out_buf.invalidate()
    # Start receive before send to reduce backpressure/stall risk.
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
    # Refresh output buffer view on CPU side.
    out_buf.invalidate()


def stop_core(core) -> None:
    # HLS control register at 0x00: clear ap_start/auto_restart.
    core.write(0x00, 0x00)


def start_core(core) -> None:
    # HLS control register at 0x00: single-shot ap_start=1.
    core.write(0x00, 0x01)


def read_gesture_npy(
    features_path: str,
    labels_path: str,
    max_samples: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load gesture windows from npy files. Supports [N,60,6] or [N,6,60] feature layouts."""
    X = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"Gesture npy must be rank-3, got shape {X.shape}")

    # Canonical internal layout is [N,60,6].
    if X.shape[1:] == (60, 6):
        Xw = X
    elif X.shape[1:] == (6, 60):
        Xw = np.transpose(X, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Gesture npy shape must be [N,60,6] or [N,6,60], got {X.shape}")

    n = min(len(Xw), len(y))
    if max_samples > 0:
        n = min(n, max_samples)
    Xw = Xw[:n]
    y = y[:n]

    keep = (y >= 0) & (y < num_classes)
    Xw = Xw[keep]
    y = y[keep]

    if len(Xw) == 0:
        raise RuntimeError("No valid gesture samples after label filtering.")
    return Xw, y


def pretty_cm(cm: np.ndarray, labels: Sequence[str]) -> None:
    # Compact text-table confusion matrix printer for terminal logs.
    header = "true\\pred | " + " ".join([f"{l[:6]:>6}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = " ".join([f"{int(cm[i, j]):6d}" for j in range(len(labels))])
        print(f"{label[:9]:>9} | {row}")


def load_gesture_labels(
    labels_file: Optional[str],
    num_classes: Optional[int],
) -> List[str]:
    """
    Resolve gesture label names in priority order:
    1) --gesture-labels-file (one label per line, index = line number)
    2) legacy defaults
    """
    labels: List[str] = []

    if labels_file:
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]

    if not labels:
        if num_classes is not None and num_classes > 0:
            labels = [f"class_{i}" for i in range(num_classes)]
        else:
            labels = list(DEFAULT_GESTURE_LABELS)

    if num_classes is not None:
        if num_classes <= 0:
            raise ValueError("--gesture-num-classes must be > 0")
        if len(labels) < num_classes:
            labels.extend([f"class_{i}" for i in range(len(labels), num_classes)])
        elif len(labels) > num_classes:
            labels = labels[:num_classes]

    return labels


def run_gesture_eval(
    gesture_core,
    gesture_dma,
    X: np.ndarray,
    y: np.ndarray,
    timeout_s: float,
    gesture_pack: str,
    num_classes: int,
) -> EvalResult:
    n = len(X)
    # Rows=true class, cols=predicted class.
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    lat_ms: List[float] = []
    correct = 0

    if gesture_pack == "q88":
        in_buffer = allocate(shape=(60 * 6,), dtype=np.uint32)
    else:
        in_buffer = allocate(shape=(60 * 6,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.uint32)

    try:
        for i in range(n):
            # Flatten in [time][channel] order, matching HLS stream loop order.
            flat = X[i].reshape(-1).astype(np.float32)  # [t][c] flatten

            if gesture_pack == "q88":
                payload = q88_pack_u32(flat)
            else:
                payload = flat
            np.copyto(in_buffer, payload)

            # End-to-end timing includes control write + DMA transfers.
            t0 = time.time()
            start_core(gesture_core)
            run_dma(gesture_dma, in_buffer, out_buffer, timeout_s=timeout_s)
            lat_ms.append((time.time() - t0) * 1000.0)
            pred = int(out_buffer[0])

            # Defensive fallback if hardware returns invalid class id.
            if pred < 0 or pred >= num_classes:
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
                # HLS expects time-major stream [t][c].
                flat = feat.T.reshape(-1).astype(np.float32)  # [t][c]
            else:
                # Optional fallback for overlays that consume channel-major stream.
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
            # Ignore samples whose labels are outside configured class range.
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
    # Build JSON-friendly aggregate metrics used by the final report.
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

    # Active default path uses npy test sets for both subsystems.
    p.add_argument("--gesture-features", default=DEFAULT_GESTURE_X)
    p.add_argument("--gesture-labels", default=DEFAULT_GESTURE_Y)
    p.add_argument("--gesture-labels-file", default=None)
    p.add_argument("--gesture-num-classes", type=int, default=None)
    p.add_argument("--voice-features", default=DEFAULT_VOICE_X)
    p.add_argument("--voice-labels", default=DEFAULT_VOICE_Y)
    p.add_argument("--gesture-max-samples", type=int, default=120)
    p.add_argument("--voice-max-samples", type=int, default=300)
    p.add_argument("--timeout-s", type=float, default=2.0)

    p.add_argument("--mode", choices=["gesture", "voice", "both"], default="both")
    # Packing mode controls how PS encodes input stream payload for each core.
    # Default is q88 for both to match hardware stream format.
    p.add_argument("--gesture-pack", choices=["q88", "float32"], default="q88")
    p.add_argument("--voice-pack", choices=["float32", "q88"], default="q88")
    # Voice feature flattening order: "tc" matches current HLS implementation.
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
    # Helpful for debugging name mismatches in CLI arguments.
    print("Available IP blocks:", ", ".join(sorted(overlay.ip_dict.keys())))

    gesture_core = getattr(overlay, args.gesture_core)
    voice_core = getattr(overlay, args.voice_core)
    gesture_dma = getattr(overlay, args.gesture_dma)
    voice_dma = getattr(overlay, args.voice_dma)

    # Start from known clean state.
    stop_core(gesture_core)
    stop_core(voice_core)

    reset_dma(gesture_dma)
    # Avoid double-reset if both names point to the same DMA object.
    if voice_dma is not gesture_dma:
        reset_dma(voice_dma)

    report: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
    }

    try:
        if args.mode in ("gesture", "both"):
            if not args.gesture_features or not args.gesture_labels:
                raise ValueError("Gesture evaluation requires --gesture-features and --gesture-labels (.npy).")
            gesture_labels = load_gesture_labels(
                labels_file=args.gesture_labels_file,
                num_classes=args.gesture_num_classes,
            )
            num_gesture_classes = len(gesture_labels)

            Xg, yg = read_gesture_npy(
                features_path=args.gesture_features,
                labels_path=args.gesture_labels,
                max_samples=args.gesture_max_samples,
                num_classes=num_gesture_classes,
            )
            print(f"\nRunning gesture test: {len(Xg)} samples")
            gres = run_gesture_eval(
                gesture_core=gesture_core,
                gesture_dma=gesture_dma,
                X=Xg,
                y=yg,
                timeout_s=args.timeout_s,
                gesture_pack=args.gesture_pack,
                num_classes=num_gesture_classes,
            )
            gsum = summarize(gres)
            print("Gesture accuracy:", f"{gsum['accuracy_pct']:.2f}%")
            pretty_cm(gres.cm, gesture_labels)
            report["gesture"] = {
                "summary": gsum,
                "labels": gesture_labels,
                "confusion_matrix": gres.cm.tolist(),
            }
            # Stop unused core before moving to next workload.
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
        # Always force-stop cores on exit/error to avoid stale running state.
        stop_core(gesture_core)
        stop_core(voice_core)

    summary_path = out_dir / "summary.json"
    # Store one artifact containing config + metrics + confusion matrices.
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {summary_path}")


if __name__ == "__main__":
    main()
