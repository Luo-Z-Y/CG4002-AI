# 🏟️ Pokémon AR Arena — Standalone AI Component
**Lead Architect:** Zhiyang  
**Project Phase:** Week 3 — Infrastructure & Simulation (Factory Build)  
**Target Platform:** Xilinx Ultra96-V2 (Zynq UltraScale+ MPSoC)

## 📋 Executive Summary
This repository contains the standalone AI processing unit for the Pokémon AR Arena. By offloading gesture and voice recognition to the FPGA fabric (PL), inference becomes deterministic and microsecond-scale, keeping the AR experience responsive and lag-free.

## 🏗️ System Architecture
The AI component operates as a **dual-engine IP core**, processing motion and audio streams in parallel with minimal CPU overhead.

### 1. Motion Engine (`gesture_referee`)
* **Input:** 6-axis IMU (accel/gyro) streamed via DMA.
* **Model:** 1D Convolutional Neural Network (1D-CNN).
* **Classes:** Swing, Thrust, Chop, Stir.
* **Optimizations:** Pipelined intake for 50 Hz real-time sampling.

### 2. Voice Engine (`voice_processor`)
* **Input:** MFCC audio features (40 coefficients × 50 windows).
* **Model:** Multi-Layer Perceptron (MLP) keyword spotter.
* **Classes:** Charizard, Blastoise, Venusaur.
* **Optimizations:** Parallel matrix multiplication via loop unrolling.

---

## 🛠️ Hardware Optimization Strategy
HLS (High-Level Synthesis) directives used to maximize Ultra96 performance:

| Pragma | Purpose | Result |
| :--- | :--- | :--- |
| `#pragma HLS PIPELINE` | Overlaps loop iterations | II=1; new sample accepted every clock cycle (100 MHz). |
| `#pragma HLS UNROLL` | Duplicates physical math units | Processes entire filter windows in a single cycle. |
| `ap_fixed<16, 8>` | 16-bit fixed-point math | 50% less BRAM usage vs float; maps to DSP48 slices. |
| `TLAST Handshaking` | AXI4-Stream EOP signal | Hard-wired synchronization with the PYNQ DMA driver. |

---

## 📂 Repository Structure
```text
AI_Component_Module/
├── hls/                   # Vitis HLS Hardware Projects
│   ├── src/               # C++ Silicon Logic (Silicon "Circuitry")
│   │   ├── gesture_referee.cpp
│   │   ├── voice_processor.cpp
│   │   ├── typedefs.h     # Global fixed-point definitions
│   │   └── weights.h      # Exported weights from PyTorch
│   └── tb/                # Testbenches for C-Simulation
├── training/              # PyTorch Development (The "Brain")
│   ├── data/              # Synthetic (W3) and Real (W4) datasets
│   ├── voice_train.py     # Keyword spotting trainer
│   ├── gesture_train.py   # 1D-CNN motion trainer
│   └── weight_exporter.py # Python-to-C++ Bridge script
└── integration/           # PYNQ / Ultra96 Deployment
    └── dma_driver.py      # Python driver to feed the FPGA
```

---

## 🗓️ Roadmap
**Week 3: Infrastructure (Current)**
- [x] Establish `typedefs.h` and fixed-point bit-widths.
- [x] Build HLS intake and normalization layers (silicon skeleton).
- [x] Implement AXI-Stream TLAST management for DMA stability.
- [x] Create standalone weight-export bridge.

**Week 4: Data & Intelligence**
- [ ] Collect 100+ real samples for each Pokémon move and voice command.
- [ ] Implement Quantization-Aware Training (QAT) in PyTorch.
- [ ] Verify C-simulation with real-world sensor CSVs.

**Week 5: Deployment**
- [ ] Finalize Vivado block design and bitstream generation.
- [ ] Deploy PYNQ overlay on Ultra96.
- [ ] Benchmark end-to-end latency (target: <30 ms).

---

## 📡 Standalone Interface Spec
To integrate this module, connect the following ports:

- **Control Bus (S_AXILITE):** Trigger inference and query idle status.
- **Data Input (AXI-STREAM):** 16-bit DMA input stream.
- **Data Output (AXI-STREAM):** Returns Move_ID (0–3) or Pokemon_ID (0–2).

---

Developed by the AI Lead for the Pokémon AR Arena Project (2026).