# 🏟️ Pokémon AR Arena — Standalone AI Component

**Target Platform:** Xilinx Ultra96-V2 (Zynq UltraScale+ MPSoC)  
**Framework:** PyTorch → Vitis HLS (C++) → PYNQ (Python)

---

## 📖 Project Overview

This repository implements a complete hardware-software co-design for real-time gesture recognition on the Ultra96-V2 platform. The system uses a 1D-CNN for IMU-based gesture classification, with hardware acceleration via FPGA and a Python-based software stack for preprocessing and control.

---

## 📂 Repository Structure

```text
CG4002-AI/
├── data/                       # Datasets & Normalization Params
│   ├── txt/                    # Raw sensor logs
│   ├── augmented_imudata.csv   # Main training dataset
│   ├── resampled_imudata.csv   # Debugging data
│   ├── mean.npy                # Global mean for Z-score
│   └── std.npy                 # Global std for Z-score
├── hardware/                   # Hardware Source & Bitstreams
│   ├── gesture_cnn.cpp         # Main HLS Source
│   ├── gesture_cnn.h           # Header
│   ├── gesture_cnn_weights.h   # Baked-in Weights
│   ├── gesture_cnn_tb.cpp      # C-Simulation Testbench
│   ├── typedefs.h              # Data type definitions
│   └── gesture_cnn.xsa         # Exported Hardware Platform
├── notebooks/                  # Training & Research
│   └── train_cnn_model.ipynb   # PyTorch Training & Weight Export
├── test/                       # Deployment Code (Ultra96)
│   ├── preprocess.py           # Real-time Signal Processing Class
│   └── test.py                 # Main Driver & Accuracy Tests
└── README.md                   # Documentation
```

---

## 🤖 AI Model Overview

### Gesture CNN (Motion Engine)
- **Status:** ✅ Active, Deployed & Verified
- **Architecture:** 1D Convolutional Neural Network (PyTorch)
- **Input:** 6-axis IMU data `[Batch, 6 Axes, 60 Samples]` (window size = 60 samples ≈ 1.2s)
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
- **Status:** 🚧 In Progress
- **Architecture:** 1D CNN on MFCC features

---

## 📋 System Architecture

The system offloads neural network computations to the FPGA (PL), while the CPU (PS) handles data preprocessing and game logic.

### Hardware Design (Vivado)
1. **Gesture CNN IP:** HLS-generated core for inference
2. **AXI DMA:** High-speed data transfer between DDR and IP core
3. **Interrupt Controller (`xlconcat`):** Aggregates DMA interrupts, signals CPU when inference completes

### Software Stack (Python/PYNQ)
1. **Preprocessing (`preprocess.py`):**
    - Motion detection (energy-based)
    - Resampling to 60 time steps
    - Z-score normalization (`mean.npy`, `std.npy`)
2. **Driver (`test.py`):**
    - DMA management with timeout protection
    - Core control via AXI-Lite (`ap_start`)

---

## 🛠️ Hardware Optimization Strategy (HLS)

We use Vitis HLS directives to maximize Ultra96 performance:

| Pragma                 | Purpose                        | Result                                              |
| :--------------------- | :----------------------------- | :-------------------------------------------------- |
| `#pragma HLS PIPELINE` | Overlaps loop iterations       | **II=1**; Processes 1 sample per clock cycle        |
| `#pragma HLS UNROLL`   | Duplicates physical math units | Parallelizes convolution kernel (3x speedup)        |
| `ap_fixed<16, 8>`      | 16-bit Fixed-Point Math        | Reduces BRAM usage by 50% vs Float; avoids FPU lag  |
| `union {int i; float f}` | Bit-level Casting            | Zero-cycle conversion from AXI-Stream bits to Fixed-Point |
| `AXI-Stream`           | DMA Interface                  | High-bandwidth data transfer without CPU copy loop  |

---

## 📂 Repository Structure

## 🚀 Quick Start

### 1. Train the Model (PC/Cloud)
1. Place raw `.txt` logs in `data/txt/`.
2. Run `notebooks/train_cnn_model.ipynb` to generate:
  - `gesture_cnn_weights.h` (copy to `hardware/`)
  - `mean.npy` and `std.npy` (copy to `data/`)

### 2. HLS Synthesis (Vitis HLS)
1. Create a new project with files from `hardware/`.
2. Run **C Synthesis** and **Export RTL**.
3. Output: AXI-Stream compatible IP core.

### 3. Hardware Implementation (Vivado)
1. Import IP into Block Design.
2. Connect **AXI DMA** (SG disabled) to IP.
3. Connect DMA `mm2s_introut` and `s2mm_introut` to Zynq `pl_ps_irq0` via a Concat block.
4. Generate Bitstream and Export `.xsa`.

### 4. Deployment (Ultra96)
1. Upload `test/` folder and `.xsa` file to the board.
2. Run:
  ```bash
  sudo python3 test.py
  ```
3. The script will:
  - Run a smoke test (1 sample per class)
  - Run 300 random inferences (confusion matrix, latency report)

---

## 🗓️ Roadmap

- [x] 1D-CNN architecture (window=60)
- [x] Data pipeline: parsing, resampling, augmentation, normalization
- [x] HLS: C++ core with AXI-Stream & fixed-point
- [x] Integration: DMA interrupts & Python driver
- [x] Verification: >95% accuracy on-board
- [ ] Voice: Finalize MFCC extractor for FPGA

### 4. Deployment (Ultra96)

1. Upload `test/` folder and `.xsa` file to the board.
2. Run the verification script:
```bash
sudo python3 test.py

```


3. Script performs:
* **Smoke Test:** Validates 1 sample per class.
* **Random Test:** Runs 300 random inferences to generate a confusion matrix and latency report.



---

## 🗓️ Roadmap

* [x] **Architecture:** 1D-CNN defined (Window=60)
* [x] **Data Pipeline:** Parsing, Resampling, Augmentation, Normalization
* [x] **HLS:** C++ Core with AXI-Stream & Fixed-Point
* [x] **Integration:** DMA Interrupts & Python Driver operational
* [x] **Verification:** >95% Accuracy on test set verified on-board
* [ ] **Voice:** Finalize MFCC extractor for FPGA
