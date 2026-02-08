# 🏟️ Pokémon AR Arena — Standalone AI Component
**Target Platform:** Xilinx Ultra96-V2 (Zynq UltraScale+ MPSoC)
**Framework:** PyTorch → Vitis HLS (C++)

---

## 🤖 AI Model Overview
This repository implements two neural network models for real-time gesture and voice recognition.

### Gesture CNN (Motion Engine)
- **Status:** ✅ Active & Deployed
- **Architecture:** 1D Convolutional Neural Network (PyTorch)
- **Input:** 6-axis IMU data `[Batch, 6 Axes, 60 Samples]`
  - *Window size optimized to 60 for hardware latency*
- **Layers:**
  - Conv1D (6→16, k=3) → ReLU → MaxPool(2)
  - Conv1D (16→32, k=3) → ReLU → MaxPool(2)
  - Fully Connected (Flatten→32→6)
- **Output Classes (6):**
  1. Raise hand and hold
  2. Shaking fist (3 shakes)
  3. Vertical chop
  4. Circular stir
  5. Horizontal Swing
  6. Punch (forward thrust)

### Voice CNN (Audio Engine)
- **Status:** 🚧 KIV / In Progress
- **Architecture:** 1D Convolutional Neural Network (PyTorch)
- **Input:** MFCC features `[Batch, 40 MFCCs, 50 Time Windows]`
- **Layers:**
  - Conv1D (16 filters, k=3)
  - Global Average Pooling
  - Fully Connected (3 classes)
- **Output Classes (3):**
  1. Charizard
  2. Blastoise
  3. Venusaur

---

## 📋 System Architecture
The AI component operates as a **dual-engine IP core** on the FPGA fabric, processing motion and audio streams in parallel with minimal CPU overhead.

### Data Flow
1. **Sensor Input:** IMU (60Hz) and Mic (16kHz) data streamed via AXI-Stream
2. **Preprocessing:**
   - Gesture: Resampling to 60 samples fixed-length
   - Voice: MFCC extraction (future)
3. **Inference:** FPGA runs 16-bit fixed-point inference (DSP slices)
4. **Output:** Interrupt-based result sent to Game Logic

---

## 🛠️ Hardware Optimization Strategy (HLS)
We use Vitis HLS directives to maximize Ultra96 performance:

| Pragma                | Purpose                        | Result                                              |
|---------------------- |------------------------------- |-----------------------------------------------------|
| `#pragma HLS PIPELINE`| Overlaps loop iterations        | **II=1**; Processes 1 sample per clock cycle         |
| `#pragma HLS UNROLL`  | Duplicates physical math units  | Parallelizes convolution kernel (3x speedup)         |
| `ap_fixed<16, 8>`     | 16-bit Fixed-Point Math         | Reduces BRAM usage by 50% vs Float; avoids FPU latency|
| `union {int i; float f}`| Bit-level Casting             | Zero-cycle conversion from AXI-Stream bits to Fixed-Point|
| `AXI-Stream`          | DMA Interface                   | High-bandwidth data transfer without CPU copy loop   |

---

## 📂 Repository Structure
```text
CG4002-AI/
├── data/                      # Raw datasets
│   ├── *.txt                  # Sensor logs
│   ├── imudata.csv            # Cleaned CSV
│   └── augmented_imudata.csv  # 10x Augmented dataset (Window=60)
├── hardware/                  # HLS Source Code (C++)
│   ├── gesture_cnn.cpp        # Main IP Core logic
│   ├── gesture_cnn.h          # Function prototype
│   ├── typedefs.h             # Fixed-point & Stream types
│   └── weights.h              # Baked-in weights (Auto-generated)
├── notebooks/                 # Jupyter Notebooks
│   └── train_cnn_model.ipynb  # End-to-End: Load → Augment → Train → Export
├── test/                      # Outdated code
└── README.md                  # Project Documentation
```

---

## 🚀 Quick Start
1. **Training (PC/Cloud):**
   - Place raw .txt logs in `data/raw_logs/`
   - Run `notebooks/train_cnn_model.ipynb`
   - Parses files, augments data (10x), trains CNN, generates `weights.h`
2. **HLS Deployment (Vivado):**
   - Open Vitis HLS
   - Add all files from `hardware/` (including new `weights.h`)
   - Set Top Function: `gesture_cnn`
   - Run C Synthesis → Export RTL
3. **Integration (Vivado Block Design):**
   - Import exported IP
   - Connect `in_stream` to AXI DMA (S2MM)
   - Connect `out_stream` to AXI DMA (MM2S) or AXI GPIO

---

## 🗓️ Roadmap
- [x] Gesture: 1D-CNN Architecture defined (Window=60)
- [x] Data: Pipeline for Parsing, Resampling & Augmentation (10x)
- [x] HLS: C++ Core with AXI-Stream & Fixed-Point support
- [ ] Voice: Finalize MFCC extractor for FPGA
- [ ] Integration: Benchmarking on Ultra96-V2