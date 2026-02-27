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
- AXIS input contract (current deployed `dual_cnn.xsa`): supports both `float32` and signed `Q8.8` packed in AXIS `data[15:0]` via `--gesture-pack`
- Labels (6, project target semantics): `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`
- Training split policy: `train/test` split first, then `train/val`; validation is used for epoch monitoring/model selection, and test is held for final reporting.
- Current dataset snapshots:
  - Legacy project gesture set (txt-logs + augmentation): `data/gesture/13022026`
    - Legacy raw file-stem/class names used in notebook parser: `raise`, `shake3`, `vertical`, `circular`, `horizontal`, `punch`
  - Public compatibility dataset (UCI HAR converted to repo schema): `data/gesture/27022026`
    - Format compatibility: converted to `[60, 6]` windows with columns `measurement_id, sequence_id, label_id, gyro_x/y/z, acc_x/y/z`
    - Current 6-class labels used in this public set: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
    - Label semantics note: these differ from project target gesture names above.
- Evaluation script: `test/dual_cnn_test.py`

### Voice CNN
- Status: Architecture updated, retrained, and integrated in `dual_cnn.xsa`
- Input feature shape: MFCC `[40, 50]`
- AXIS input contract (current deployed `dual_cnn.xsa`): signed `Q8.8` packed into AXIS `data[15:0]`
- Current dataset snapshots:
  - Active training/deployment labels (`yes`, `no`, `go`): `data/audio/18022026`
  - New downloaded candidate set (`marvin`, `sheila`, `visual`): `data/audio/25022026`
- Evaluation script: `test/dual_cnn_test.py`
- Voice labels (current 3-class implementation): `go`, `no`, `yes`
- Future plan: migrate to real Pokémon labels (for first iteration: `pikachu`, `charizard`, `babusaur`) after dataset refresh and retraining.

### Router
- Runtime arbitration scaffold: `test/router.py`
- Selects gesture vs voice core using motion and voice-energy thresholds

## Key Design Decisions

1. Dual-IP architecture (`gesture_cnn` and `voice_cnn` as separate accelerators)
- Rationale: independent debug/tuning, cleaner Vivado integration, and isolated resource management.

2. Mixed input contracts in current deployed dual overlay
- Gesture path currently uses float32 AXIS input.
- Voice path currently uses Q8.8 AXIS input (`data[15:0]`).
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
- Current deployed dual overlay interface reads float32 from AXIS data.
- Outcome: gesture path is stable on board with current software packing.

### Input Normalization (Z-score)
- Z-score is used during training in both notebooks.
- Export step in both notebooks sets `fuse_input_norm=True`, which folds input normalization into `conv1` weights/biases for raw-input inference.
- Therefore, runtime PS preprocessing for the dual test does not need separate z-score for either model when using those fused exported weights.

### Latest Ultra96 Dual-IP Evidence (2026-02-20)
Command used:
```bash
python3 test/dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0 \
  --mode both \
  --gesture-pack float32 \
  --voice-pack q88
```

Observed results:
- Gesture accuracy: `92.50%` (120 samples)
- Voice accuracy: `75.00%` (300 samples)
- Evidence artifact: `report/evidence_dual/20260220_184325/summary.json`

### Latest Voice Software Test (2026-02-25)
- Dataset: `yes/no/go`, split before augmentation to avoid leakage
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
  - Gesture: `test/gesture_test.py`
  - Voice: `test/voice_test.py`
- Routing scaffold: `test/router.py`

## Quick Start

### 1. Train Models
- Gesture: run `notebooks/train_gesture_cnn.ipynb`
- Voice: run `notebooks/train_voice_cnn.ipynb`

### 2. Gesture Evaluation (Ultra96)

```bash
python3 test/dual_cnn_test.py \
  --xsa-path hardware/vivado/dual_cnn.xsa \
  --mode gesture \
  --gesture-features data/gesture/27022026/gesture_X_test.npy \
  --gesture-labels data/gesture/27022026/gesture_y_test.npy \
  --gesture-num-classes 6 \
  --gesture-norm none \
  --gesture-pack q88 \
  --save-dir report/evidence_dual \
  --tag baseline
```

### 3. Voice Evaluation (Ultra96)

```bash
python3 test/dual_cnn_test.py \
  --xsa-path hardware/vivado/dual_cnn.xsa \
  --mode voice \
  --voice-features data/audio/25022026/voice_X_test.npy \
  --voice-labels data/audio/25022026/voice_y_test.npy \
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
  --gesture-dma axi_dma_gesture \
  --voice-dma axi_dma_voice \
  --demo-mode
```

## Progress vs Deliverables

- AI model design: complete for gesture and voice
- FPGA implementation: gesture evidenced; voice IP updated and ready for final integrated validation
- Ultra96 setup and low-level access: implemented in test scripts (`Overlay`, DMA, MMIO)
- Software implementation and evaluation: complete gesture flow; voice flow implemented with fixed-point streaming path
- Hardware evidence in repo: `report/gesture_cnn_csynth.rpt`, `report/vivado-project-summary.png`
- Power-management hooks available in `test/gesture_test.py` (`--cpu-governor`, `--pl-clock-mhz`, `--power-w`)

## Evaluation Metrics for Final Report

1. Fit and integration metrics
- Per-IP LUT/FF/BRAM/DSP and post-integration total (both IPs + DMA + AXI interconnect + control).
- Keep practical LUT headroom; avoid targeting near-maximum LUT in HLS-only estimates.

2. Timing closure metrics
- Use post-Vivado WNS as pass/fail:
  - WNS >= 0: target clock met
  - WNS < 0: timing violation

3. End-to-end runtime on PYNQ
- Measure PS preprocessing, DMA send, IP compute, DMA receive, and total inference latency.

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
