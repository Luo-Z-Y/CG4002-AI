from pynq import Overlay, allocate
import numpy as np
import pandas as pd
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from preprocess import GesturePreprocessor

# ============================================================
# CONFIG
# ============================================================
XSA_PATH = "gesture_cnn_updated.xsa"
CSV_PATH = "augmented_imudata.csv"

LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
C = len(LABELS)

# Your CSV columns include sequence_id; always sort by it to match time order
TIME_COL = "sequence_id"
FEATURE_COLS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]

# Test modes
RUN_SMOKE_TEST = True          # 1 sample per class
RUN_RANDOM_TEST = True         # random test over many windows
SEED = 42
N_RANDOM = 300                 # max 300 in your dataset
DMA_TIMEOUT_S = 2.0
DEFAULT_SAVE_DIR = "evidence"


# ============================================================
# DMA HELPERS
# ============================================================
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
        mmio.write(off + 0x00, 0x4)  # reset
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x1)  # run
    time.sleep(0.01)

def start_hls_cores(overlay):
    started = []
    for name, meta in overlay.ip_dict.items():
        if "hls" in str(meta.get("type", "")).lower():
            try:
                core = getattr(overlay, name)
                core.write(0x00, 0x81)  # ap_start=1 + auto_restart=1
                started.append(name)
            except Exception as e:
                print(f"⚠️ Could not start {name}: {e}")
    if started:
        print("Started HLS cores:", started)
    else:
        print("⚠️ No HLS cores auto-started.")

def run_dma_with_timeout(dma, in_buf, out_buf, timeout_s=2.0):
    in_buf.flush()
    out_buf.invalidate()

    # RX first reduces stall risk
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t0 = time.time()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.time() - t0 > timeout_s:
            print("❌ DMA TIMEOUT")
            dump_dma(dma)
            raise TimeoutError("DMA stalled")
        time.sleep(0.001)

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()


# ============================================================
# REPORTING
# ============================================================
def pretty_confusion(cm):
    header = "true\\pred | " + " ".join([f"{l[:5]:>5}" for l in LABELS])
    line = "-" * len(header)
    print("\nConfusion Matrix (counts):")
    print(header)
    print(line)
    for i, l in enumerate(LABELS):
        row = " ".join([f"{cm[i, j]:5d}" for j in range(C)])
        print(f"{l[:9]:>9} | {row}")


def get_cpu_governor():
    gov_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_file.exists():
        try:
            return gov_file.read_text().strip()
        except Exception:
            return "unknown"
    return "unavailable"


def save_artifacts(run_dir, summary, cm=None, lat_ms=None):
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "summary.json"
    summary_txt = run_dir / "summary.txt"

    summary_json.write_text(json.dumps(summary, indent=2))

    lines = [
        f"Run ID: {summary['run_id']}",
        f"Timestamp: {summary['timestamp']}",
        f"XSA: {summary['config']['xsa_path']}",
        f"CSV: {summary['config']['csv_path']}",
        f"Seed: {summary['config']['seed']}",
        f"N random: {summary['config']['n_random']}",
        f"CPU governor: {summary['system']['cpu_governor']}",
        f"PL clock (MHz): {summary['system']['pl_clock_mhz']}",
        f"Power (W): {summary['system']['power_w']}",
        f"Random accuracy (%): {summary['results']['random_accuracy_pct']}",
        f"Latency mean/p50/p90/p99 (ms): "
        f"{summary['results']['latency_mean_ms']} / "
        f"{summary['results']['latency_p50_ms']} / "
        f"{summary['results']['latency_p90_ms']} / "
        f"{summary['results']['latency_p99_ms']}",
    ]
    summary_txt.write_text("\n".join(lines) + "\n")

    if cm is not None:
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        cm_df.to_csv(run_dir / "confusion_matrix.csv", index=True, index_label="true\\pred")

    if lat_ms is not None and len(lat_ms) > 0:
        pd.DataFrame({"latency_ms": lat_ms}).to_csv(run_dir / "latency_samples.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Ultra96 gesture CNN hardware evaluation script")
    parser.add_argument("--xsa-path", default=XSA_PATH, help="Path to .xsa/.bit container for Overlay")
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to evaluation CSV")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--n-random", type=int, default=N_RANDOM, help="Number of random test windows")
    parser.add_argument("--timeout-s", type=float, default=DMA_TIMEOUT_S, help="DMA timeout seconds")
    parser.add_argument("--no-smoke", action="store_true", help="Disable smoke test")
    parser.add_argument("--no-random", action="store_true", help="Disable random test")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory to save report evidence files")
    parser.add_argument("--tag", default="", help="Optional short tag appended to run folder name")
    parser.add_argument("--power-w", type=float, default=None, help="Measured total power in watts")
    parser.add_argument("--pl-clock-mhz", type=float, default=None, help="PL clock in MHz used for this run")
    parser.add_argument("--cpu-governor", default=None, help="CPU governor label (auto-detected if omitted)")
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    run_smoke = not args.no_smoke
    run_random = not args.no_random

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp if not args.tag else f"{timestamp}_{args.tag}"
    run_dir = Path(args.save_dir) / run_id

    print(f"Programming FPGA with {args.xsa_path}...")
    overlay = Overlay(args.xsa_path)
    dma = overlay.axi_dma_0

    reset_dma(dma)
    start_hls_cores(overlay)

    in_buffer = allocate(shape=(360,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.int32)

    # Your preprocess.py now has fixed mean/std inside the class
    pre = GesturePreprocessor(fs=50, max_len=100, target_len=60)

    print("✅ FPGA Ready.")
    print("\n--- Loading CSV ---")
    df = pd.read_csv(args.csv_path)

    # Ensure numeric ordering
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    # Build valid windows (exactly 60 rows, 60 unique sequence_id)
    valid = []
    grouped = df.groupby("measurement_id", sort=False)
    per_class = {i: 0 for i in range(C)}

    for mid, g in grouped:
        if len(g) != 60:
            continue
        if g[TIME_COL].isna().any():
            continue
        if g["label_id"].isna().any():
            continue
        if len(np.unique(g[TIME_COL].to_numpy())) != 60:
            continue
        y = int(g.iloc[0]["label_id"])
        if 0 <= y < C:
            valid.append((mid, y))
            per_class[y] += 1

    print(f"Valid windows: {len(valid)}")
    print("Per-class available:", {LABELS[k]: v for k, v in per_class.items()})
    if not valid:
        print("❌ No valid windows found.")
        return

    # Fast index for group rows
    mid_to_idx = df.groupby("measurement_id", sort=False).indices

    try:
        smoke_results = []
        cm = None
        lat_ms = []
        random_accuracy_pct = None
        random_total = 0
        random_correct = 0
        random_total_time_s = None

        # -----------------------------
        # SMOKE TEST (1 per class)
        # -----------------------------
        if run_smoke:
            print("\n--- Smoke Test (1 sample per class) ---")
            found = set()

            for mid, y_true in valid:
                if y_true in found:
                    continue

                rows = mid_to_idx[mid]
                g = df.iloc[rows].sort_values(TIME_COL, ascending=True, kind="mergesort")
                raw = g[FEATURE_COLS].to_numpy(dtype=np.float32)  # (60,6)

                # IMPORTANT: apply your fixed train mean/std from preprocess.py
                x = pre._zscore(raw)  # (60,6) float32

                flat = x.reshape(-1).astype(np.float32)  # (360,)
                np.copyto(in_buffer, flat)

                try:
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)
                except TimeoutError:
                    reset_dma(dma)
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)

                pred = int(out_buffer[0])
                status = "✅ PASS" if pred == y_true else f"❌ FAIL (Got {LABELS[pred] if 0 <= pred < C else pred})"
                print(f"Class {y_true} ({LABELS[y_true]}): {status}")
                smoke_results.append(
                    {
                        "measurement_id": int(mid),
                        "true_class": int(y_true),
                        "pred_class": int(pred),
                        "pass": bool(pred == y_true),
                    }
                )

                found.add(y_true)
                if len(found) == C:
                    break

        # -----------------------------
        # RANDOM TEST (confusion matrix)
        # -----------------------------
        if run_random:
            rng = np.random.default_rng(args.seed)
            n_test = min(args.n_random, len(valid))
            chosen = rng.choice(len(valid), size=n_test, replace=False)

            cm = np.zeros((C, C), dtype=np.int32)
            correct = 0
            lat_ms = []

            print(f"\n--- Random Test: {n_test} samples (seed={args.seed}) ---")
            t_all0 = time.time()

            for k, idx in enumerate(chosen, 1):
                mid, y_true = valid[int(idx)]
                rows = mid_to_idx[mid]
                g = df.iloc[rows].sort_values(TIME_COL, ascending=True, kind="mergesort")
                raw = g[FEATURE_COLS].to_numpy(dtype=np.float32)  # (60,6)

                x = pre._zscore(raw)
                flat = x.reshape(-1).astype(np.float32)
                np.copyto(in_buffer, flat)

                t0 = time.time()
                try:
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=args.timeout_s)
                except TimeoutError:
                    reset_dma(dma)
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=args.timeout_s)
                lat_ms.append((time.time() - t0) * 1000.0)

                pred = int(out_buffer[0])
                if pred < 0 or pred >= C:
                    pred = 0

                cm[y_true, pred] += 1
                correct += int(pred == y_true)

                if k % 25 == 0 or k == n_test:
                    print(f"Progress {k:4d}/{n_test}: acc={100.0*correct/k:5.1f}%  last={lat_ms[-1]:6.2f} ms")

            t_all1 = time.time()
            acc = 100.0 * correct / n_test
            random_accuracy_pct = float(acc)
            random_total = int(n_test)
            random_correct = int(correct)
            random_total_time_s = float(t_all1 - t_all0)

            print("\n=== RANDOM TEST RESULTS ===")
            print(f"Total tested: {n_test}")
            print(f"Correct:      {correct}")
            print(f"Accuracy:     {acc:.2f}%")
            print(f"Total time:   {(t_all1 - t_all0):.2f} s")

            if lat_ms:
                lat = np.array(lat_ms, dtype=np.float64)
                print("\nLatency (ms) per inference:")
                print(
                    f"  mean={lat.mean():.2f}  p50={np.percentile(lat,50):.2f}  "
                    f"p90={np.percentile(lat,90):.2f}  p99={np.percentile(lat,99):.2f}  max={lat.max():.2f}"
                )

            print("\nPer-class accuracy:")
            for i in range(C):
                tot = cm[i].sum()
                ok = cm[i, i]
                pct = 100.0 * ok / tot if tot else 0.0
                print(f"  {i} {LABELS[i]:>5}: {ok:4d}/{tot:4d}  ({pct:5.1f}%)")

            pretty_confusion(cm)

        lat_arr = np.array(lat_ms, dtype=np.float64) if len(lat_ms) > 0 else None
        cpu_governor = args.cpu_governor if args.cpu_governor else get_cpu_governor()
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "xsa_path": args.xsa_path,
                "csv_path": args.csv_path,
                "seed": int(args.seed),
                "n_random": int(args.n_random),
                "timeout_s": float(args.timeout_s),
                "run_smoke": bool(run_smoke),
                "run_random": bool(run_random),
            },
            "system": {
                "cpu_governor": cpu_governor,
                "pl_clock_mhz": args.pl_clock_mhz,
                "power_w": args.power_w,
            },
            "results": {
                "valid_windows": int(len(valid)),
                "smoke_passed": int(sum(1 for r in smoke_results if r["pass"])),
                "smoke_total": int(len(smoke_results)),
                "random_total": random_total,
                "random_correct": random_correct,
                "random_accuracy_pct": random_accuracy_pct,
                "random_total_time_s": random_total_time_s,
                "latency_mean_ms": float(lat_arr.mean()) if lat_arr is not None else None,
                "latency_p50_ms": float(np.percentile(lat_arr, 50)) if lat_arr is not None else None,
                "latency_p90_ms": float(np.percentile(lat_arr, 90)) if lat_arr is not None else None,
                "latency_p99_ms": float(np.percentile(lat_arr, 99)) if lat_arr is not None else None,
                "latency_max_ms": float(lat_arr.max()) if lat_arr is not None else None,
            },
            "smoke_results": smoke_results,
        }
        save_artifacts(run_dir, summary, cm=cm, lat_ms=lat_ms)
        print(f"\n📁 Saved evidence to: {run_dir.resolve()}")

    finally:
        # Cleanup (PYNQ buffers use .close(), not .free())
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass
        del in_buffer, out_buffer


if __name__ == "__main__":
    main()
