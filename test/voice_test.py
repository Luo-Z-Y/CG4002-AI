from pynq import Overlay, allocate
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ==============================
# Defaults
# ==============================
XSA_PATH = "voice_cnn_updated.xsa"
FEATURES_NPY = "voice_X_test.npy"   # expected shape [N, 40, 50]
LABELS_NPY = "voice_y_test.npy"     # expected shape [N]
LABELS = ["Class0", "Class1", "Class2"]
DMA_TIMEOUT_S = 2.0


# ==============================
# DMA helpers
# ==============================
def dump_dma(dma):
    for ch, name in [(dma.sendchannel, "MM2S"), (dma.recvchannel, "S2MM")]:
        mmio = ch._mmio
        off = ch._offset
        ctrl = mmio.read(off + 0x00)
        stat = mmio.read(off + 0x04)
        print(f"{name} CTRL={ctrl:#010x} STAT={stat:#010x}")


def reset_dma(dma):
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x4)
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x1)
    time.sleep(0.01)


def start_hls_cores(overlay):
    started = []
    for name, meta in overlay.ip_dict.items():
        if "hls" in str(meta.get("type", "")).lower():
            try:
                core = getattr(overlay, name)
                core.write(0x00, 0x81)  # ap_start + auto_restart
                started.append(name)
            except Exception as e:
                print(f"Could not start {name}: {e}")
    if started:
        print("Started HLS cores:", started)


def run_dma_with_timeout(dma, in_buf, out_buf, timeout_s=2.0):
    in_buf.flush()
    out_buf.invalidate()

    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t0 = time.time()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.time() - t0 > timeout_s:
            print("DMA TIMEOUT")
            dump_dma(dma)
            raise TimeoutError("DMA stalled")
        time.sleep(0.001)

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()


# ==============================
# Utils
# ==============================
def pretty_confusion(cm, labels):
    header = "true\\pred | " + " ".join([f"{l[:6]:>6}" for l in labels])
    line = "-" * len(header)
    print("\nConfusion Matrix (counts):")
    print(header)
    print(line)
    for i, l in enumerate(labels):
        row = " ".join([f"{cm[i, j]:6d}" for j in range(len(labels))])
        print(f"{l[:9]:>9} | {row}")


def save_artifacts(out_dir, summary, cm, lat_ms, labels):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Run ID: {summary['run_id']}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Accuracy (%): {summary['results']['accuracy_pct']:.2f}\n")
        f.write(
            "Latency mean/p50/p90/p99/max (ms): "
            f"{summary['results']['latency_mean_ms']:.2f}/"
            f"{summary['results']['latency_p50_ms']:.2f}/"
            f"{summary['results']['latency_p90_ms']:.2f}/"
            f"{summary['results']['latency_p99_ms']:.2f}/"
            f"{summary['results']['latency_max_ms']:.2f}\n"
        )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True, index_label="true\\pred")
    pd.DataFrame({"latency_ms": lat_ms}).to_csv(out_dir / "latency_samples.csv", index=False)


# ==============================
# Main
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Voice CNN Ultra96 evaluator")
    p.add_argument("--xsa-path", default=XSA_PATH)
    p.add_argument("--features-npy", default=FEATURES_NPY)
    p.add_argument("--labels-npy", default=LABELS_NPY)
    p.add_argument("--timeout-s", type=float, default=DMA_TIMEOUT_S)
    p.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    p.add_argument("--save-dir", default="report/evidence_voice")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = ts if not args.tag else f"{ts}_{args.tag}"
    run_dir = Path(args.save_dir) / run_id

    print(f"Programming FPGA with {args.xsa_path}...")
    overlay = Overlay(args.xsa_path)
    dma = overlay.axi_dma_0

    reset_dma(dma)
    start_hls_cores(overlay)

    X = np.load(args.features_npy).astype(np.float32)
    y = np.load(args.labels_npy).astype(np.int64)

    if X.ndim != 3 or X.shape[1] != 40 or X.shape[2] != 50:
        raise ValueError(f"Expected X shape [N,40,50], got {X.shape}")
    if len(X) != len(y):
        raise ValueError("Features/labels length mismatch")

    n_classes = int(max(np.max(y) + 1, len(LABELS)))
    labels = LABELS[:n_classes]
    if len(labels) < n_classes:
        labels.extend([f"Class{i}" for i in range(len(labels), n_classes)])

    n = len(X)
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    in_buffer = allocate(shape=(40 * 50,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.int32)

    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    lat_ms = []
    correct = 0

    try:
        print(f"Running {n} voice inferences...")
        t_all0 = time.time()

        for i in range(n):
            feat = X[i]  # [40,50]
            flat = feat.reshape(-1).astype(np.float32)
            np.copyto(in_buffer, flat)

            t0 = time.time()
            try:
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=args.timeout_s)
            except TimeoutError:
                reset_dma(dma)
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=args.timeout_s)

            lat_ms.append((time.time() - t0) * 1000.0)
            pred = int(out_buffer[0])

            if pred < 0 or pred >= n_classes:
                pred = 0

            y_true = int(y[i])
            if y_true < 0 or y_true >= n_classes:
                continue

            cm[y_true, pred] += 1
            correct += int(pred == y_true)

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"Progress {i+1:4d}/{n}: acc={100.0*correct/(i+1):5.1f}% last={lat_ms[-1]:6.2f} ms")

        total_time = time.time() - t_all0

        lat = np.array(lat_ms, dtype=np.float64)
        acc = 100.0 * correct / max(n, 1)

        print("\n=== VOICE TEST RESULTS ===")
        print(f"Total tested: {n}")
        print(f"Correct:      {correct}")
        print(f"Accuracy:     {acc:.2f}%")
        print(f"Total time:   {total_time:.2f} s")
        print("\nLatency (ms):")
        print(
            f"  mean={lat.mean():.2f} p50={np.percentile(lat,50):.2f} "
            f"p90={np.percentile(lat,90):.2f} p99={np.percentile(lat,99):.2f} max={lat.max():.2f}"
        )

        pretty_confusion(cm, labels)

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "xsa_path": args.xsa_path,
                "features_npy": args.features_npy,
                "labels_npy": args.labels_npy,
                "max_samples": int(args.max_samples),
                "timeout_s": float(args.timeout_s),
            },
            "results": {
                "total_tested": int(n),
                "correct": int(correct),
                "accuracy_pct": float(acc),
                "total_time_s": float(total_time),
                "latency_mean_ms": float(lat.mean()),
                "latency_p50_ms": float(np.percentile(lat, 50)),
                "latency_p90_ms": float(np.percentile(lat, 90)),
                "latency_p99_ms": float(np.percentile(lat, 99)),
                "latency_max_ms": float(lat.max()),
            },
        }

        save_artifacts(run_dir, summary, cm, lat_ms, labels)
        print(f"\nSaved evidence to: {run_dir.resolve()}")

    finally:
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
