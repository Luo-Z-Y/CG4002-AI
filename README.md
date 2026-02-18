# CG4002-AI

Standalone AI component for Ultra96-V2 (Zynq UltraScale+ MPSoC).

Framework path: PyTorch -> Vitis HLS (C++) -> Vivado -> PYNQ (Python)

## Project Overview

This repository implements a hardware-software co-design for real-time gesture recognition on the Ultra96-V2 platform. The system uses a 1D-CNN for IMU-based gesture classification, with hardware acceleration via FPGA and a Python-based software stack for preprocessing and control. A voice pipeline is also included for training and software-side routing, but voice FPGA deployment is not yet hardware-validated.

## Current Status (2026-02-18)

### Gesture CNN
- Status: Active, hardware-integrated, and evaluable on Ultra96
- Input: IMU window `[60, 6]` (`gyro_x/y/z`, `acc_x/y/z`)
- Labels (6): `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`
- Evaluation script: `test/gesture_test.py`

### Voice CNN
- Status: Dataset + preprocessing + training + HLS + eval script ready; full on-board validation still pending
- Input feature shape: MFCC `[40, 50]`
- Current dataset snapshot: `data/audio/18022026`
- Evaluation script: `test/voice_test.py`
- Voice manifest (`data/audio/18022026/voice_manifest.csv`):
  - `go`: 1000
  - `no`: 1000
  - `yes`: 1000

### Router
- Runtime arbitration scaffold: `test/router.py`
- Selects gesture vs voice core using motion and voice-energy thresholds

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
- HLS IP cores for gesture and voice inference
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
python3 test/gesture_test.py \
  --xsa-path hardware/gesture_cnn.xsa \
  --csv-path data/gesture/13022026/augmented_imudata.csv \
  --save-dir report/evidence_gesture \
  --tag baseline
```

### 3. Voice Evaluation (Ultra96)

```bash
python3 test/voice_test.py \
  --xsa-path <voice_overlay.xsa> \
  --features-npy data/audio/18022026/voice_X_test.npy \
  --labels-npy data/audio/18022026/voice_y_test.npy \
  --save-dir report/evidence_voice \
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
- FPGA implementation: complete and evidenced for gesture; voice final board validation pending
- Ultra96 setup and low-level access: implemented in test scripts (`Overlay`, DMA, MMIO)
- Software implementation and evaluation: complete gesture flow; voice flow implemented and awaiting final hardware evidence
- Hardware evidence in repo: `report/gesture_cnn_csynth.rpt`, `report/vivado-project-summary.png`
- Power-management hooks available in `test/gesture_test.py` (`--cpu-governor`, `--pl-clock-mhz`, `--power-w`)

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
