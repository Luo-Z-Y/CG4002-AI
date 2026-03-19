from pynq import Overlay, allocate
import numpy as np
import pandas as pd
import time

# ============================================================
# CONFIG
# ============================================================
XSA_PATH = "gesture_cnn_updated.xsa"
CSV_PATH = "augmented_imudata.csv"

LABELS = ["0", "1", "2", "3", "4", "5"]

# If your CSV has a time/order column inside each measurement_id group,
# set it here (examples: "timestamp", "sample_idx"). Otherwise leave None.
TIME_COL = None  # e.g. "timestamp"

# Feature order MUST match training / HLS expectation.
# Current order (gyro then acc):
FEATURE_COLS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
# If you suspect mismatch, try this instead:
# FEATURE_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# If your model expects normalized inputs, set NORMALIZE=True and provide mean/std.
NORMALIZE = False
MEAN_PATH = "mean.npy"  # shape (6,) or (360,)
STD_PATH  = "std.npy"   # shape (6,) or (360,)

# Output words: if your IP outputs >1 word, increase this (e.g., 16) to avoid stalls.
OUT_WORDS = 1

# DMA timeout (seconds)
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
    # reset both channels (useful after Ctrl+C / previous error)
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x4)  # reset bit
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x1)  # run
    time.sleep(0.01)

def start_hls_cores(overlay):
    # Start any HLS cores (ap_start=1 + auto_restart=1) so TREADY is asserted.
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
        print("⚠️ No HLS cores auto-started (maybe ap_ctrl_none or not HLS).")

def run_dma_with_timeout(dma, in_buf, out_buf, timeout_s=2.0):
    # cache maintenance
    in_buf.flush()
    out_buf.invalidate()

    # IMPORTANT: arm RX first to reduce backpressure stalls
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t0 = time.time()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.time() - t0 > timeout_s:
            print("❌ DMA TIMEOUT (likely backpressure / core not running / size mismatch).")
            dump_dma(dma)
            raise TimeoutError("DMA stalled")
        time.sleep(0.001)

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Programming FPGA with {XSA_PATH}...")
    overlay = Overlay(XSA_PATH)
    dma = overlay.axi_dma_0

    # Reset DMA and start compute cores
    reset_dma(dma)
    start_hls_cores(overlay)

    # Allocate buffers
    in_buffer = allocate(shape=(360,), dtype=np.float32)
    out_buffer = allocate(shape=(OUT_WORDS,), dtype=np.int32)

    # Load normalization (optional)
    mean = std = None
    if NORMALIZE:
        mean = np.load(MEAN_PATH).astype(np.float32)
        std  = np.load(STD_PATH).astype(np.float32)
        print(f"Loaded mean/std: mean{mean.shape}, std{std.shape}")

    print("✅ FPGA Ready.")
    print("\n--- Starting Smoke Test (1 sample per class) ---")

    df = pd.read_csv(CSV_PATH)

    found_classes = set()
    grouped = df.groupby("measurement_id")

    try:
        for measure_id, group in grouped:
            label_id = int(group.iloc[0]["label_id"])

            if label_id in found_classes:
                continue

            # Ensure consistent order within a measurement window
            if TIME_COL is not None and TIME_COL in group.columns:
                group = group.sort_values(TIME_COL, ascending=True)

            raw_matrix = group[FEATURE_COLS].to_numpy(dtype=np.float32)
            if len(raw_matrix) != 60:
                continue

            # Optional normalization
            if NORMALIZE:
                if mean.shape == (6,) and std.shape == (6,):
                    raw_matrix = (raw_matrix - mean) / (std + 1e-6)
                elif mean.shape == (360,) and std.shape == (360,):
                    flat_tmp = raw_matrix.flatten()
                    flat_tmp = (flat_tmp - mean) / (std + 1e-6)
                    raw_matrix = flat_tmp.reshape(60, 6)
                else:
                    raise ValueError(f"Unsupported mean/std shape: {mean.shape} / {std.shape}")

            flat_data = raw_matrix.flatten().astype(np.float32)
            np.copyto(in_buffer, flat_data)

            # Run FPGA (with one retry after DMA reset)
            try:
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)
            except TimeoutError:
                print("Retrying once after DMA reset...")
                reset_dma(dma)
                run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=DMA_TIMEOUT_S)

            pred = int(out_buffer[0])
            if 0 <= pred < len(LABELS):
                pred_name = LABELS[pred]
            else:
                pred_name = str(pred)

            status = "✅ PASS" if pred == label_id else f"❌ FAIL (Got {pred_name})"
            print(f"Class {label_id} ({LABELS[label_id]}): {status}")

            found_classes.add(label_id)
            if len(found_classes) == 6:
                break

        print("\nSmoke Test Complete.")

    finally:
        # Cleanup (PYNQ versions differ; .free() may not exist)
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass
        del in_buffer, out_buffer


if __name__ == "__main__":
    main()
