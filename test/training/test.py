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

# Allocate Buffers (360 floats -> 1 int)
in_buffer = allocate(shape=(360,), dtype=np.float32)
out_buffer = allocate(shape=(1,), dtype=np.int32)

print("✅ FPGA Ready.")

# ==========================================
# 2. SMOKE TEST (1 Sample Per Class)
# ==========================================
print("\n--- Starting Smoke Test (1 sample per class) ---")

# Load Data
df = pd.read_csv(CSV_PATH)
LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]

# We will just pick the first occurrence of each label_id (0 to 5)
found_classes = set()
grouped = df.groupby('measurement_id')

for measure_id, group in grouped:
    label_id = group.iloc[0]['label_id']
    
    # If we already tested this class, skip it
    if label_id in found_classes:
        continue
        
    # Prepare Data
    raw_matrix = group[['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']].values
    if len(raw_matrix) != 60: continue
    
    # Flatten & Copy
    flat_data = raw_matrix.flatten().astype(np.float32)
    np.copyto(in_buffer, flat_data)
    
    # Cache maintenance
    in_buffer.flush()
    out_buffer.invalidate()

    # Run FPGA
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.transfer(in_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    
    out_buffer.invalidate()

    # Check Result
    pred = int(out_buffer[0])
    status = "✅ PASS" if pred == label_id else f"❌ FAIL (Got {LABELS[pred]})"
    
    print(f"Class {label_id} ({LABELS[label_id]}): {status}")
    
    found_classes.add(label_id)
    
    # Stop if we found all 6
    if len(found_classes) == 6:
        break

print("\nSmoke Test Complete.")
# Cleanup
in_buffer.free()
out_buffer.free()