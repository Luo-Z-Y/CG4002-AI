from pynq import Overlay, allocate
import numpy as np
import pandas as pd
import time

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


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Programming FPGA with {XSA_PATH}...")
    overlay = Overlay(XSA_PATH)
    dma = overlay.axi_dma_0

    reset_dma(dma)
    start_hls_cores(overlay)

    in_buffer = allocate(shape=(360,), dtype=np.float32)
    out_buffer = allocate(shape=(1,), dtype=np.int32)

    # Your preprocess.py now has fixed mean/std inside the class
    pre = GesturePreprocessor(fs=50, max_len=100, target_len=60)

    print("✅ FPGA Ready.")
    print("\n--- Loading CSV ---")
    df = pd.read_csv(CSV_PATH)

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
        # -----------------------------
        # SMOKE TEST (1 per class)
        # -----------------------------
        if RUN_SMOKE_TEST:
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

                found.add(y_true)
                if len(found) == C:
                    break

        # -----------------------------
        # RANDOM TEST (confusion matrix)
        # -----------------------------
        if RUN_RANDOM_TEST:
            rng = np.random.default_rng(SEED)
            n_test = min(N_RANDOM, len(valid))
            chosen = rng.choice(len(valid), size=n_test, replace=False)

            cm = np.zeros((C, C), dtype=np.int32)
            correct = 0
            lat_ms = []

            print(f"\n--- Random Test: {n_test} samples (seed={SEED}) ---")
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
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)
                except TimeoutError:
                    reset_dma(dma)
                    run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)
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