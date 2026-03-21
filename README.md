# CG4002-AI

Standalone AI component for Ultra96-V2 (Zynq UltraScale+ MPSoC).

Framework path: PyTorch -> Vitis HLS (C++) -> Vivado -> PYNQ (Python)

## Project Overview

This repository implements a hardware-software co-design for real-time gesture and voice recognition on Ultra96/Ultra96-V2. The deployment path is PyTorch training/export on PS, fixed-point streaming over AXI DMA, and FPGA inference with HLS IP cores integrated in Vivado.

## Target Board and PL Budget

- Board: Ultra96 (96Boards CE) / Ultra96-V2
- SoC: Zynq UltraScale+ MPSoC ZU3EG A484
- Programmable Logic (ZU3EG) limits:
  - LUT: 70,560
  - FF: 141,120
  - BRAM: 216
  - DSP: 360

## Board / Constraint References

- Ultra96 product page (board part and key platform specs)
- Ultra96-V2 product brief (hardware overview)
- Avnet Ultra96-PYNQ repository (Vivado/XDC examples)
- Xilinx IIoT-SPYN Ultra96 constraints examples

## Current Status (2026-02-27)

### Gesture CNN
- Status: Active and hardware-integrated in `dual_cnn.xsa`
- Input: IMU window `[60, 6]` (`gyro_x/y/z`, `acc_x/y/z`)
- AXIS input contract (current deployed `dual_cnn.xsa`): signed `Q8.8` packed in AXIS `data[15:0]` (default in `ultra96/dual_cnn_test.py`)
- Active labels (6, current public dataset): `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
- Project-target gesture meanings: `0=Raise`, `1=Shake`, `2=Chop`, `3=Stir`, `4=Swing`, `5=Punch`
- Deployment/runtime output convention: gesture labels are emitted as numeric strings `0` to `5`
- Training split policy: `train/test` split first, then `train/val`; validation is used for epoch monitoring/model selection, and test is held for final reporting.
- Current dataset snapshots:
  - Legacy project gesture set (txt-logs + augmentation): `data/gesture/20260213`
    - Legacy raw file-stem/class names used in notebook parser: `raise`, `shake3`, `vertical`, `circular`, `horizontal`, `punch`
  - Public compatibility dataset (UCI HAR converted to repo schema): `data/gesture/20260227`
    - Format compatibility: converted to `[60, 6]` windows with columns `measurement_id, sequence_id, label_id, gyro_x/y/z, acc_x/y/z`
    - Current 6-class labels used in this public set: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
    - Label semantics note: these differ from project-target gesture names above.
- Evaluation script: `ultra96/dual_cnn_test.py`

### Voice CNN
- Status: Architecture updated, retrained, and integrated in `dual_cnn.xsa`
- Input feature shape: MFCC `[40, 50]`
- AXIS input contract (current deployed `dual_cnn.xsa`): signed `Q8.8` packed into AXIS `data[15:0]`
- Current dataset snapshots:
  - Active Pokemon label set (`bulbasaur`, `charizard`, `pikachu`): `data/audio/20260321`
  - Previous SpeechCommands-derived label set (`marvin`, `sheila`, `visual`): `data/audio/20260225`
  - Legacy label set (`yes`, `no`, `go`): `data/audio/20260218`
- Evaluation script: `ultra96/dual_cnn_test.py`
- Voice label meanings (current 3-class implementation): `0=bulbasaur`, `1=charizard`, `2=pikachu`
- Deployment/runtime output convention: voice labels are emitted as numeric strings `0` to `2`
- Dataset note: the latest Pokemon folder now contains `55` clips per class after slicing two additional 5-utterance raw recordings for each label.

### Router
- Runtime arbitration scaffold: `test/router.py`
- Selects gesture vs voice core using motion and voice-energy thresholds

## Key Design Decisions

1. Dual-IP architecture (`gesture_cnn` and `voice_cnn` as separate accelerators)
- Rationale: independent debug/tuning, cleaner Vivado integration, and isolated resource management.

2. Input contract in current deployed dual overlay
- Gesture path uses Q8.8 AXIS input (`data[15:0]`).
- Voice path uses Q8.8 AXIS input (`data[15:0]`).
- Rationale: this matches the currently deployed and validated `dual_cnn.xsa`.

2a. Current HLS IP input expectation (source-of-truth in `hardware/*.cpp`)
- Both `gesture_cnn` and `voice_cnn` consume signed `Q8.8` packed in AXIS `data[15:0]`.
- The IPs interpret stream payload via bit-cast (`q88_from_axis`) into `ap_fixed<16,8>` and do fixed-point inference internally.
- Therefore float-to-fixed quantization/packing is expected on PS before DMA; the IPs do not perform float-input quantization.

3. Quantization for voice
- Voice quantization is done on PS (Python/NumPy) before DMA.
- Rationale: reduces PL-side conversion logic and LUT pressure.

4. Explicit 32-bit output class encoding
- Decision: cast class output to `ap_uint<32>`.
- Rationale: avoids AXI data-width ambiguity and keeps interface contracts explicit.

## Progress Summary

### Voice IP
- PS packs MFCC float features to Q8.8 int16 before DMA.
- PL consumes fixed-point stream directly (no float union conversion).
- Observed outcome: LUT usage reduced to a board-safe range compared with prior near-limit runs; DSP usage remains moderate; latency remains in few-thousand cycles (tens of microseconds at ~100 MHz).

### Gesture IP
- PS packs IMU float features to Q8.8 int16 before DMA.
- PL consumes fixed-point stream directly from AXIS `data[15:0]`.
- Outcome: gesture path is stable on board with current software packing.

### Input Normalization (Z-score)
- Z-score is used during training in both notebooks.
- Export step in both notebooks sets `fuse_input_norm=True`, which folds input normalization into `conv1` weights/biases for raw-input inference.
- Therefore, runtime PS preprocessing for the dual test does not need separate z-score for either model when using those fused exported weights.

### Latest Ultra96 Dual-IP Evidence (2026-02-20)
Command used:
```bash
python3 ultra96/dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0 \
  --mode both \
  --gesture-pack q88 \
  --voice-pack q88
```

Observed results:
- Gesture accuracy: `92.50%` (120 samples)
- Voice accuracy: `75.00%` (300 samples)
- Evidence artifact: `report/evidence_dual/20260220_184325/summary.json`

### Latest Voice Software Test (2026-02-25)
- Dataset: `marvin/sheila/visual`, split before augmentation to avoid leakage
- Test accuracy (float): `77.28%`
- Test accuracy (Q8.8-sim): `77.28%`
- Test set size: `1800` samples (`600` per class)

Why Ultra96 accuracy is slightly lower than notebook accuracy:
- Hardware path introduces finite precision effects (`ap_fixed<16,8>` arithmetic and Q8.8 input quantization for voice), while notebook training/eval is typically float.
- PS-PL deployment path includes additional implementation details not present in notebook inference (packing, AXIS ordering, cast/saturation behavior).
- The board test uses the real deployed overlay and end-to-end DMA/IP execution, so it captures integration noise and quantization mismatch that notebook-only evaluation does not.

## Repository Structure (Authoritative)

```text
CG4002-AI/
|-- README.md
|-- data/
|   |-- gesture/
|   `-- voice/
|-- hardware/
|   |-- gesture_cnn.cpp
|   |-- gesture_cnn.h
|   |-- gesture_typedefs.h
|   |-- gesture_cnn_weights.h
|   |-- gesture_cnn_tb.cpp
|   |-- gesture_cnn.xsa
|   |-- voice_cnn.cpp
|   |-- voice_cnn.h
|   |-- voice_typedefs.h
|   |-- voice_cnn_weights.h
|   `-- voice_cnn_tb.cpp
|-- notebooks/
|   |-- train_gesture_cnn.ipynb
|   `-- train_voice_cnn.ipynb
|-- test/
|   |-- gesture_preprocess.py
|   |-- gesture_test.py
|   |-- voice_preprocess.py
|   |-- voice_test.py
|   |-- router.py
|   `-- outdated/
`-- report/
    |-- gesture_cnn_csynth.rpt
    `-- vivado-project-summary.png
```

## Architecture Summary

### Hardware
- Separate HLS IP cores for gesture and voice inference
- AXI DMA for PS-PL transfers
- AXI-Lite control for core start/stop and MMIO interactions

### Software
- Gesture preprocessing: `test/gesture_preprocess.py`
- Voice preprocessing: `test/voice_preprocess.py`
- Evaluation and artifact export:
  - Gesture: `ultra96/dual_cnn_test.py --mode gesture`
  - Voice: `ultra96/dual_cnn_test.py --mode voice`
- Routing scaffold: `test/router.py`

## Quick Start

### 1. Train Models
- Gesture: run `notebooks/train_gesture_cnn.ipynb`
- Voice: run `notebooks/train_voice_cnn.ipynb`

### 2. Gesture Evaluation (Ultra96)

```bash
python3 ultra96/dual_cnn_test.py \
  --xsa-path hardware/vivado/dual_cnn.xsa \
  --mode gesture \
  --gesture-features data/gesture/20260227/gesture_X_test.npy \
  --gesture-labels data/gesture/20260227/gesture_y_test.npy \
  --gesture-num-classes 6 \
  --gesture-pack q88 \
  --save-dir report/evidence_dual \
  --tag baseline
```

### 3. Voice Evaluation (Ultra96)

```bash
python3 ultra96/dual_cnn_test.py \
  --xsa-path hardware/vivado/dual_cnn.xsa \
  --mode voice \
  --voice-features data/audio/20260321/voice_X_test.npy \
  --voice-labels data/audio/20260321/voice_y_test.npy \
  --voice-pack q88 \
  --save-dir report/evidence_dual \
  --tag baseline
```

### 4. Dual-IP Router (Scaffold)

```bash
python3 test/router.py \
  --xsa-path <dual_overlay.xsa> \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0 \
  --demo-mode
```

## Progress vs Deliverables

- AI model design: complete for gesture and voice
- FPGA implementation: gesture evidenced; voice IP updated and ready for final integrated validation
- Ultra96 setup and low-level access: implemented in test scripts (`Overlay`, DMA, MMIO)
- Software implementation and evaluation: complete gesture flow; voice flow implemented with fixed-point streaming path
- Hardware evidence in repo: `report/gesture_cnn_csynth.rpt`, `report/vivado-project-summary.png`
- Power-management hooks available in `ultra96/dual_cnn_test.py` (`--cpu-governor`, `--cpu-freq-khz`, `--pl-clock-mhz`, `--power-w`, `--power-sysfs-path`)

## Ultra96 Power/Timing Flags (`ultra96/dual_cnn_test.py`)

Each flag below is independent and can be used together in one command.

- `--cpu-governor <name>`
  - Purpose: request Linux CPU governor (for example `performance`, `userspace`, `ondemand`).
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --cpu-governor performance`
  - Note: typically needs root privileges.

- `--cpu-freq-khz <khz>`
  - Purpose: request CPU clock frequency in kHz (for A53 cpufreq policy).
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --cpu-governor userspace --cpu-freq-khz 1200000`
  - Note: usually requires `userspace` governor and root privileges.

- `--pl-clock-mhz <mhz>`
  - Purpose: request PL FCLK0 clock via PYNQ `Clocks` API.
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --pl-clock-mhz 100`
  - Note: applied best-effort; depends on board/overlay support.

- `--power-w <watts>`
  - Purpose: provide manual board power to compute per-inference energy estimates in report.
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --power-w 4.2`

- `--power-sysfs-path <path>`
  - Purpose: read board power value from sysfs-like file and store in runtime report.
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --power-sysfs-path /sys/class/hwmon/hwmon0/power1_input`

- `--power-sysfs-scale <scale>`
  - Purpose: scale raw value from `--power-sysfs-path` into Watts.
  - Example: `python3 ultra96/dual_cnn_test.py --mode both --power-sysfs-path /sys/class/hwmon/hwmon0/power1_input --power-sysfs-scale 1e-6`
  - Note: use `1e-6` when the sysfs value is in microwatts.

Recorded timing/overhead fields in `summary.json` (for both gesture and voice summaries):

- `latency_total_*_ms`: end-to-end per sample (`prep + control + DMA path`).
- `latency_inference_*_ms`: inference path on PS side (`control + DMA path`).
- `latency_comm_*_ms`: DMA communication overhead (`submit + wait`).
- `latency_prep_*_ms`: preprocessing/packing overhead on PS.
- `latency_dma_submit_*_ms`: DMA transfer submission overhead.
- `latency_dma_wait_*_ms`: DMA wait/complete overhead.
- `energy_total_mean_mj`, `energy_inference_mean_mj`, `energy_comm_mean_mj`: present when `power_w` is available.
- `runtime_controls`: captures requested values, before/after observed values, and warnings when settings cannot be applied.

## Evaluation Metrics for Final Report

1. Fit and integration metrics
- Per-IP LUT/FF/BRAM/DSP and post-integration total (both IPs + DMA + AXI interconnect + control).
- Keep practical LUT headroom; avoid targeting near-maximum LUT in HLS-only estimates.

2. Timing closure metrics
- Use post-Vivado WNS as pass/fail:
  - WNS >= 0: target clock met
  - WNS < 0: timing violation

3. End-to-end runtime on PYNQ
- Measure and report PS preprocessing overhead, control overhead, DMA communication overhead, inference-path latency, and total end-to-end latency.

4. Accuracy stability across deployment stages
- Compare:
  - PyTorch float model
  - Python-side quantized-input simulation (Q8.8)
  - FPGA on-board classification

## Remaining Integration Work

1. Complete final Vivado block design with both IPs + DMA(s) + AXI interconnect.
2. Verify full design utilization against ZU3EG PL limits.
3. Close timing at target PL clock (non-negative WNS).
4. Run end-to-end dual-model PYNQ tests with latency and accuracy evidence capture.

## Notes

- `test/outdated/` contains legacy experiments kept for reference only.
- If dataset locations change, update CLI paths accordingly.

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
