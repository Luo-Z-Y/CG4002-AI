# 🏟️ Pokémon AR Arena — Standalone AI Component

**Target Platform:** Xilinx Ultra96-V2 (Zynq UltraScale+ MPSoC)  
**Framework:** PyTorch → Vitis HLS (C++) → PYNQ (Python)

---

## 📖 Project Overview

This repository implements a hardware-software co-design for real-time gesture recognition on the Ultra96-V2 platform. The system uses a 1D-CNN for IMU-based gesture classification, with hardware acceleration via FPGA and a Python-based software stack for preprocessing and control. A voice pipeline is also included for training and software-side routing, but voice FPGA deployment is not yet hardware-validated.

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
│   ├── voice_cnn.cpp           # Voice HLS Source (scaffold)
│   ├── voice_cnn.h             # Voice Header
│   ├── voice_cnn_weights.h     # Voice Weights Header (export target)
│   ├── voice_cnn_tb.cpp        # Voice C-Simulation Testbench
│   ├── voice_typedefs.h        # Voice Data type definitions
│   └── gesture_cnn.xsa         # Exported Hardware Platform
├── notebooks/                  # Training & Research
│   ├── train_gesture_cnn.ipynb # Gesture PyTorch Training & Weight Export
│   └── train_voice_cnn.ipynb   # Voice PyTorch Training & Weight Export
├── test/                       # Deployment Code (Ultra96)
│   ├── preprocess.py           # Real-time Signal Processing Class
│   ├── test.py                 # Main Driver & Accuracy Tests
│   ├── voice_preprocess.py     # Voice feature extraction (MFCC)
│   ├── voice_test.py           # Voice Driver & Accuracy Tests
│   └── router.py               # Real-time gesture/voice routing scaffold
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
- **Status:** ⚠️ Software pipeline ready, FPGA deployment not yet validated
- **Architecture:** 1D CNN on MFCC features
- **Current scope:** Training notebook + preprocessing + runtime router scaffold
- **Important:** Voice hardware files are provided as integration scaffolding and are not yet verified on-board as a production voice accelerator.

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
2. Run `notebooks/train_gesture_cnn.ipynb` to generate:
  - `gesture_cnn_weights.h` (copy to `hardware/`)
  - `mean.npy` and `std.npy` (copy to `data/`)

### 1.1 Train Voice Model (Software Pipeline)
1. Place voice `.wav` files under `data/audio/<class_name>/`.
2. Run `notebooks/train_voice_cnn.ipynb` to generate:
  - `voice_cnn_weights.h` (optional export target for future HLS integration)
  - voice feature splits (`voice_X_train.npy`, `voice_X_test.npy`, `voice_y_train.npy`, `voice_y_test.npy`)
  - `voice_mean.npy` and `voice_std.npy`
3. Current limitation:
  - Voice FPGA inference is not yet hardware-validated in this project.

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
1. Upload the `test/` folder and `.xsa` file to the board.
2. Before running the test script, set up the environment as follows:
  ```bash
  sudo su -
  source /usr/local/share/pynq-venv/bin/activate
  export XILINX_XRT=/usr
  cd /home/xilinx/cg4002_test
  ```
  - `sudo su -`: Switches to the root user (required for DMA access).
  - `source /usr/local/share/pynq-venv/bin/activate`: Activates the PYNQ Python environment.
  - `export XILINX_XRT=/usr`: Ensures Xilinx Runtime (XRT) is available for FPGA communication.
  - `cd /home/xilinx/cg4002_test`: Change to your project directory (adjust if needed).
3. Now run the test script:
  ```bash
  python3 test.py --tag baseline_perf --cpu-governor performance --pl-clock-mhz 100 --save-dir report/evidence
  ```
  - `--tag baseline_perf`: Labels this run (used in output folder name) so you can distinguish baseline vs power-saving experiments.
  - `--cpu-governor performance`: Records the CPU governor setting used for this run in the saved summary files.
  - `--pl-clock-mhz 100`: Records the programmable-logic clock (MHz) used for this run in the saved summary files.
  - `--save-dir report/evidence`: Writes all generated evidence files into `report/evidence/<timestamp>_<tag>/`.

  Optional flags:
  - `--power-w <value>`: Attach measured board/on-chip power (W) to the run summary for power-management comparisons.
  - `--n-random <N>`: Number of random evaluation windows (default: 300).
  - `--seed <int>`: Random seed for reproducible window selection (default: 42).
  - `--timeout-s <seconds>`: DMA timeout threshold per inference (default: 2.0s).
  - `--xsa-path <file>`: Path to the hardware overlay file (default set in script).
  - `--csv-path <file>`: Path to evaluation CSV dataset (default set in script).
  - `--no-smoke`: Skip 1-sample-per-class smoke test.
  - `--no-random`: Skip random-test evaluation and only run smoke test.
4. The script will:
  - Run a smoke test (1 sample per class)
  - Run 300 random inferences (confusion matrix, latency report)
  - Save report-ready evidence files (`summary.json`, `summary.txt`, `confusion_matrix.csv`, `latency_samples.csv`)

### 5. Real-Time Router (Gesture vs Voice)
Use `test/router.py` to arbitrate which IP should run in real time based on motion score and voice energy score.

Example:
```bash
python3 test/router.py \
  --xsa-path dual_ai.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_gesture \
  --voice-dma axi_dma_voice \
  --motion-thr 1.2 \
  --voice-thr 0.015 \
  --priority gesture \
  --demo-mode
```

Notes:
- Router starts only one selected core per dispatch and keeps the other stopped.
- Replace demo/synthetic input hooks with real IMU + audio sources before production use.
- Voice routing support exists in software; full voice FPGA validation is still pending.

---

## ✅ Assessment Criteria and Current Progress

This section maps the assessment rubric to current repository progress.

Status legend:
- `✅ Completed`
- `🟡 In Progress / Partially Completed`
- `❌ Not Yet Completed`

### 1) AI Model Design (Video)
- `✅` Neural network type documented for gesture (`1D CNN`) and voice (`1D CNN on MFCC`).
- `✅` Number/type of layers documented for gesture and implemented in code.
- `✅` Number of neurons/channels documented (gesture fully; voice in notebook/model code).
- `✅` Activation functions documented and implemented (`ReLU`).
- Evidence:
  - `notebooks/train_gesture_cnn.ipynb`
  - `notebooks/train_voice_cnn.ipynb`
  - `hardware/gesture_cnn.cpp`
  - `hardware/voice_cnn.cpp`

### 2) FPGA Implementation (Video)
- `✅` Gesture hardware implementation per layer exists in HLS C++.
- `✅` Mathematical basis represented in code: convolution, bias, ReLU, pooling, dense, argmax.
- `✅` Activation function implementation exists in HLS (`relu` helper).
- `🟡` Voice HLS implementation scaffold exists but is not hardware-validated on board.
- Evidence:
  - `hardware/gesture_cnn.cpp`
  - `hardware/voice_cnn.cpp`

### 3) Ultra96 Setup (Live + Video)
- `✅` NN implementation approach documented (HLS C++ -> Vivado IP -> PYNQ).
- `✅` IP import into Vivado documented in quick-start flow.
- `✅` Interface usage documented and implemented (`AXIS`, `AXI-Lite`, DMA MMIO reads/writes).
- `✅` FPGA programming flow shown in script (`Overlay(...)`) and README.
- `✅` Low-level register access implemented (`mmio.write/mmio.read` for DMA + core control).
- `🟡` Voice Ultra96 deployment path exists (`test/voice_test.py`, `test/router.py`) but full voice hardware validation pending.
- Evidence:
  - `README.md`
  - `test/test.py`
  - `test/voice_test.py`
  - `test/router.py`

### 4) Ultra96 Simulation Setup (Video)
- `✅` Gesture IP verification tool and testbench provided (`gesture_cnn_tb.cpp`).
- `✅` Voice IP testbench scaffold provided (`voice_cnn_tb.cpp`).
- `🟡` Need to present/record final simulation outputs in report/video artifacts.
- Evidence:
  - `hardware/gesture_cnn_tb.cpp`
  - `hardware/voice_cnn_tb.cpp`

### 5) Software Implementation and Evaluation (Live + Video)
- `✅` Gesture segmentation/feature selection/parameters implemented (`preprocess.py`).
- `✅` Gesture model training pipeline implemented (`train_gesture_cnn.ipynb`).
- `✅` Voice software training pipeline implemented (`train_voice_cnn.ipynb`).
- `✅` Libraries/software used are documented in notebook and code (`PyTorch`, `torchaudio`, `NumPy`, `pandas`, `scikit-learn`, `PYNQ`).
- `✅` Validation path implemented (train/test split, classification metrics, confusion matrix).
- `✅` Hardware test script saves report-ready artifacts for gesture.
- Evidence:
  - `test/preprocess.py`
  - `notebooks/train_gesture_cnn.ipynb`
  - `notebooks/train_voice_cnn.ipynb`
  - `test/test.py`

### 6) Hardware Accelerator Implementation and Evaluation (Live + Video)
- `✅` Gesture accelerator evaluation flow implemented:
  - Accuracy
  - Confusion matrix
  - Latency statistics (mean/p50/p90/p99/max)
  - Saved evidence files for report
- `✅` Resource/timing/power reports added under `report/` for gesture.
- `🟡` Direct software-vs-hardware comparison table still needs to be finalized in report text.
- `🟡` Voice hardware accelerator evaluation not complete (software + scaffold only).
- Evidence:
  - `test/test.py`
  - `report/gesture_cnn_csynth.rpt`
  - `report/vivado-project-summary.png`

### 7) AI FPGA and Ultra96 Power Management (Video)
- `🟡` Measurement hooks are present in gesture test script (`--cpu-governor`, `--pl-clock-mhz`, `--power-w`) to record controlled experiments.
- `❌` Full documented experiment results (before/after tables for governor/PL clock/peripherals/cores) are not fully completed in repository yet.
- Suggested completion:
  - Run baseline vs powersave vs lower PL clock.
  - Capture latency + power evidence files and summarize in report.
- Evidence:
  - `test/test.py`

### Submission Note (Important)
- Gesture pipeline is the primary completed hardware path.
- Voice pipeline is currently:
  - `✅` software training + preprocessing + routing scaffold
  - `🟡` HLS code scaffold
  - `❌` not yet fully hardware-validated on Ultra96 for final claims

---

## Original Assessment Text (Reference)

AI Model Design  (Video)
Describe the setup of the AI model(s) implemented. For example, but not limited to Neural network type
Number and type of layers
Number of neurons
Activation function used

FPGA Implementation  (Video)
Describe the hardware implementation of the neural network for every layer, including The mathematical basis for each layer and their functions
Implementation of the activation functions

Ultra96 Setup  (Live + Video)
Implementation of the neural network (e.g. using C++ within Vivado High-Level Synthesis (HLS) tools, or directly using RTL)
Importing Intellectual Property (IP) core into Vivado
The input/output interface (AXIS/MMIO etc)
Explain and demonstrate how to program the FPGA using the generated bitstream
Explain how low-level read and writes are handled to the FPGA

Ultra96 Simulation Setup  (Video)
Tool(s) used to verify the IP Core
Description of the simulation results

Software Implementation and Evaluation  (Live + Video)
Explain how you segment, select features, and parameters.
Training of the neural network
Indicate the libraries and software you used for classification
Model validation
Confusion matrix and classification accuracy

Hardware Accelerator Implementation and Evaluation  (Live + Video)
Evaluation of the hardware accelerator. For example, but not limited to Accuracy as compared to the software implementation
Execution time, including inference and communication overhead
Hardware Resource Usage (LUTs, RAMs, DSP, FF etc.)
Describe and outline the optimizations implemented. For example, but not limited to Fixed Point Arithmetic, Loop Unrolling, Dataflow Pipelining.

AI FPGA and Ultra96 - Power Management  (Video)
Describe and outline the system power management methods implemented for both the Ultra96 board as well as the FPGA. For example, but not limited to Changing CPU clock frequency
Changing programmable logic clock rate
Disabling Peripherals
Switching off CPU cores
