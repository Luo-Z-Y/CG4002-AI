from pynq import Overlay, allocate
import numpy as np
import pandas as pd
import time

# ==========================================
# 1. SETUP FPGA (No Reboot Needed)
# ==========================================
XSA_PATH = "gesture_cnn_updated.xsa"
CSV_PATH = "augmented_imudata.csv"

print(f"Programming FPGA with {XSA_PATH}...")
overlay = Overlay(XSA_PATH)
dma = overlay.axi_dma_0

def dump_dma(dma):
    # AXI DMA regs: 0x00 CTRL, 0x04 STATUS (per channel)
    for ch, name in [(dma.sendchannel, "MM2S"), (dma.recvchannel, "S2MM")]:
        mmio = ch._mmio
        off = ch._offset
        ctrl = mmio.read(off + 0x00)
        stat = mmio.read(off + 0x04)
        print(f"{name} CTRL={ctrl:#010x} STAT={stat:#010x}")

def reset_dma(dma):
    # Put both channels into reset, then run
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x4)   # reset bit
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        mmio = ch._mmio
        off = ch._offset
        mmio.write(off + 0x00, 0x1)   # run/stop = 1
    time.sleep(0.01)

def start_hls_cores(overlay):
    """
    Many HLS IPs need ap_start=1 before they accept AXIS input (assert TREADY).
    We'll start any IP whose 'type' contains 'hls' (safe filter).
    """
    started = []
    for name, meta in overlay.ip_dict.items():
        ip_type = str(meta.get("type", "")).lower()
        if "hls" in ip_type:
            try:
                core = getattr(overlay, name)
                core.write(0x00, 0x81)  # ap_start=1, auto_restart=1
                started.append(name)
            except Exception as e:
                print(f"⚠️  Could not start {name}: {e}")
    if started:
        print("Started HLS cores:", started)
    else:
        print("⚠️  No HLS cores auto-started (maybe your design uses ap_ctrl_none or no HLS IP).")

def run_dma_with_timeout(dma, in_buf, out_buf, timeout_s=2.0):
    # Cache maintenance
    in_buf.flush()
    out_buf.invalidate()

    # ARM RX FIRST (important for backpressure-heavy pipelines)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t0 = time.time()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.time() - t0 > timeout_s:
            print("❌ DMA TIMEOUT (stream likely backpressured / core not running / size mismatch).")
            dump_dma(dma)
            raise TimeoutError("DMA stalled")

        time.sleep(0.001)

    # Final waits (should return immediately after idle)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()

# Reset DMA + start compute cores
reset_dma(dma)
start_hls_cores(overlay)

# Allocate buffers
in_buffer = allocate(shape=(360,), dtype=np.float32)

# IMPORTANT: keep as (1,) if your IP outputs class-id only.
# If you still stall, change to (16,) temporarily to test output-size mismatch.
out_buffer = allocate(shape=(1,), dtype=np.int32)

print("✅ FPGA Ready.")

# ==========================================
# 2. SMOKE TEST (1 Sample Per Class)
# ==========================================
print("\n--- Starting Smoke Test (1 sample per class) ---")

df = pd.read_csv(CSV_PATH)
LABELS = ["0", "1", "2", "3", "4", "5"]

found_classes = set()
grouped = df.groupby('measurement_id')

for measure_id, group in grouped:
    label_id = int(group.iloc[0]['label_id'])

    if label_id in found_classes:
        continue

    raw_matrix = group[['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']].values
    if len(raw_matrix) != 60:
        continue

    flat_data = raw_matrix.flatten().astype(np.float32)
    np.copyto(in_buffer, flat_data)

    try:
        run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=2.0)
    except TimeoutError:
        print("Attempting DMA reset and retry once...")
        reset_dma(dma)
        # retry once
        run_dma_with_timeout(dma, in_buffer, out_buffer, timeout_s=2.0)

    pred = int(out_buffer[0])
    status = "✅ PASS" if pred == label_id else f"❌ FAIL (Got {LABELS[pred] if 0 <= pred < len(LABELS) else pred})"
    print(f"Class {label_id} ({LABELS[label_id]}): {status}")

    found_classes.add(label_id)
    if len(found_classes) == 6:
        break

print("\nSmoke Test Complete.")

in_buffer.free()
out_buffer.free()
