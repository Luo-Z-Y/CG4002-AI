from pynq import Overlay, allocate
import numpy as np
import pandas as pd
import time

# ============================================================
# CONFIG
# ============================================================
XSA_PATH = "gesture_cnn_updated.xsa"
CSV_PATH = "augmented_imudata.csv"

LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
NUM_CLASSES = len(LABELS)

# If you have an ordering column inside each measurement_id group, set it:
TIME_COL = None  # e.g. "timestamp" or "sample_idx"

# Feature order MUST match training / HLS expectation
FEATURE_COLS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
# Try this if you suspect feature order mismatch:
# FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Normalization (only enable if your training used it and you have mean/std)
NORMALIZE = False
MEAN_PATH = "mean.npy"  # shape (6,) or (360,)
STD_PATH  = "std.npy"   # shape (6,) or (360,)

# If your IP outputs more than 1 word, increase this (e.g., 16)
OUT_WORDS = 1

# Random test settings
SEED = 42
TOTAL_RANDOM_TESTS = 300          # total number of random windows to test
PER_CLASS_CAP = None              # e.g., 100 to limit each class to <=100 samples; None = no cap
DMA_TIMEOUT_S = 2.0


# ============================================================
# DMA / IP HELPERS
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
        ip_type = str(meta.get("type", "")).lower()
        if "hls" in ip_type:
            try:
                core = getattr(overlay, name)
                core.write(0x00, 0x81)  # ap_start=1, auto_restart=1
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

def pretty_confusion(cm, labels):
    # cm: (C,C)
    header = "true\\pred | " + " ".join([f"{l[:5]:>5}" for l in labels])
    line = "-" * len(header)
    print("\nConfusion Matrix (counts):")
    print(header)
    print(line)
    for i, l in enumerate(labels):
        row = " ".join([f"{cm[i, j]:5d}" for j in range(cm.shape[1])])
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
    out_buffer = allocate(shape=(OUT_WORDS,), dtype=np.int32)

    mean = std = None
    if NORMALIZE:
        mean = np.load(MEAN_PATH).astype(np.float32)
        std  = np.load(STD_PATH).astype(np.float32)
        print(f"Loaded mean/std: mean{mean.shape}, std{std.shape}")

    print("✅ FPGA Ready.")
    print("\n--- Building valid windows list ---")

    df = pd.read_csv(CSV_PATH)
    grouped = df.groupby("measurement_id")

    # Build a list of valid measurement_ids with label_id
    # (We will fetch group data on demand to avoid storing all raw windows.)
    valid = []
    per_class_count = {i: 0 for i in range(NUM_CLASSES)}

    for mid, g in grouped:
        if len(g) != 60:
            continue
        lid = int(g.iloc[0]["label_id"])
        if lid < 0 or lid >= NUM_CLASSES:
            continue
        if PER_CLASS_CAP is not None and per_class_count[lid] >= PER_CLASS_CAP:
            continue
        valid.append((mid, lid))
        per_class_count[lid] += 1

    if len(valid) == 0:
        print("❌ No valid 60-row windows found.")
        return

    print(f"Valid windows: {len(valid)}")
    print("Per-class available:", {LABELS[k]: v for k, v in per_class_count.items()})

    # Random sample
    rng = np.random.default_rng(SEED)
    n_test = min(TOTAL_RANDOM_TESTS, len(valid))
    chosen_idx = rng.choice(len(valid), size=n_test, replace=False)

    # Prepare stats
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    correct = 0
    total = 0
    latencies_ms = []

    # For fast lookup: groupby get_group can be slow, so index rows by measurement_id
    # We'll build an index map from measurement_id -> row indices once.
    mid_to_idx = df.groupby("measurement_id").indices

    print(f"\n--- Random Test: {n_test} samples (seed={SEED}) ---")
    t_all0 = time.time()

    try:
        for k, idx in enumerate(chosen_idx, 1):
            mid, true_id = valid[int(idx)]

            rows = mid_to_idx[mid]
            g = df.iloc[rows]

            if TIME_COL is not None and TIME_COL in g.columns:
                g = g.sort_values(TIME_COL, ascending=True)

            raw_matrix = g[FEATURE_COLS].to_numpy(dtype=np.float32)  # (60,6)

            # Optional normalization
            if NORMALIZE:
                if mean.shape == (6,) and std.shape == (6,):
                    raw_matrix = (raw_matrix - mean) / (std + 1e-6)
                elif mean.shape == (360,) and std.shape == (360,):
                    flat_tmp = raw_matrix.flatten()
                    flat_tmp = (flat_tmp - mean) / (std + 1e-6)
                    raw_matrix = flat_tmp.reshape(60, 6)
                else:
                    raise ValueError(f"Unsupported mean/std shape: {mean.shape}/{std.shape}")

            flat = raw_matrix.flatten().astype(np.float32)
            np.copyto(in_buffer, flat)

            t0 = time.time()
            try:
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)
            except TimeoutError:
                # retry once after reset
                reset_dma(dma)
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)

            dt_ms = (time.time() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            pred = int(out_buffer[0])
            # guard
            if pred < 0 or pred >= NUM_CLASSES:
                pred = 0  # treat as 0 to keep cm valid (or handle differently)

            cm[true_id, pred] += 1
            total += 1
            correct += int(pred == true_id)

            # lightweight progress
            if k % 25 == 0 or k == n_test:
                acc = 100.0 * correct / total
                print(f"Progress {k:4d}/{n_test}: acc={acc:5.1f}%  last={dt_ms:6.2f} ms")

    finally:
        # Cleanup (avoid .free(); not present in some PYNQ)
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass
        del in_buffer, out_buffer

    t_all1 = time.time()
    acc = 100.0 * correct / max(1, total)

    print("\n=== RANDOM TEST RESULTS ===")
    print(f"Total tested: {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {acc:.2f}%")
    print(f"Total time:   {(t_all1 - t_all0):.2f} s")

    if latencies_ms:
        lat = np.array(latencies_ms, dtype=np.float64)
        print("\nLatency (ms) per inference:")
        print(f"  mean={lat.mean():.2f}  p50={np.percentile(lat,50):.2f}  p90={np.percentile(lat,90):.2f}  p99={np.percentile(lat,99):.2f}  max={lat.max():.2f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(NUM_CLASSES):
        tot_i = cm[i].sum()
        ok_i = cm[i, i]
        pct = 100.0 * ok_i / tot_i if tot_i > 0 else 0.0
        print(f"  {i} {LABELS[i]:>5}: {ok_i:4d}/{tot_i:4d}  ({pct:5.1f}%)")

    pretty_confusion(cm, LABELS)


if __name__ == "__main__":
    main()