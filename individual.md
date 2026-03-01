# Individual Assessment Answers (CG4002-AI)

This file answers all assessment prompts directly, based on the current implementation in this repository (`hls/`, `notebooks/`, `ultra96/`).

## 1) AI Model Design (Video)

### Model setup
Two separate 1D CNN models are implemented:

1. Gesture CNN (`notebooks/train_gesture_cnn.ipynb`, `hls/gesture_cnn.cpp`)
2. Voice CNN (`notebooks/train_voice_cnn.ipynb`, `hls/voice_cnn.cpp`)

Both are temporal models deployed as fixed-point HLS IP.

### Gesture CNN architecture
- Input: `[6, 60]` (6 IMU channels, 60 timesteps)
- Layer 1: `Conv1d(6 -> 16, k=3, padding=1)` + `ReLU` + `MaxPool1d(2)`
- Layer 2: `Conv1d(16 -> 32, k=3, padding=1)` + `ReLU` + `MaxPool1d(2)`
- Flatten: `32 x 15 = 480`
- FC1: `Linear(480 -> 32)` + `ReLU`
- FC2: `Linear(32 -> 6)`
- Output: Argmax over 6 classes

Active gesture labels (public dataset mode):
- `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

### Voice CNN architecture
- Input: `[40, 50]` MFCC tensor (40 coefficients, 50 frames)
- Layer 1: `Conv1d(40 -> 16, k=3, padding=1)` + `ReLU` + `MaxPool1d(2)` (`50 -> 25`)
- Layer 2: `Conv1d(16 -> 32, k=3, padding=1)` + `ReLU` + `AdaptiveAvgPool1d(1)`
- FC: `Linear(32 -> 3)`
- Output: Argmax over 3 classes

Active voice labels:
- `marvin`, `sheila`, `visual`

### Activation functions used
- ReLU in hidden layers for both models
- No softmax in hardware output path (argmax on logits)

### Notes on normalization and quantization policy
- Training uses z-score normalization.
- Deployment folds z-score into the first convolution (`fuse_input_norm=True`) during weight export.
- Runtime board input is raw features; both IPs consume Q8.8 packed data.

---

## 2) FPGA Implementation (Video)

### Mathematical basis per layer
For both models, layer computations are standard:

- 1D convolution:
  - `y[o,t] = b[o] + sum_i sum_k x[i, t+k-p] * w[o,i,k]`
- ReLU:
  - `relu(x) = max(0, x)`
- Max pooling (gesture + voice block1):
  - `y[o,t] = max(x[o,2t], x[o,2t+1])`
- Global/adaptive average pooling (voice block2):
  - `y[o] = (1/T) * sum_t x[o,t]` with `T=25`
- Fully connected:
  - `z[c] = b[c] + sum_i h[i] * w[c,i]`
- Final classification:
  - `argmax(z)`

### Hardware data types and precision
- Main tensor type: `ap_fixed<16,8, AP_TRN, AP_SAT>` (Q8.8)
- AXIS packet type: `ap_axiu<32,0,0,0>`
- Packed data contract: signed Q8.8 value in `data[15:0]` of each AXIS word

### Activation implementation
- ReLU implemented explicitly in C++ (`x > 0 ? x : 0`)

### Implementation style in HLS
- Top-level functions:
  - `gesture_cnn(hls::stream<axis_t>&, hls::stream<axis_t>&)`
  - `voice_cnn(hls::stream<axis_t>&, hls::stream<axis_t>&)`
- Both use:
  - AXIS input/output interfaces
  - AXI-Lite control (`s_axilite port=return`)

### Key optimization/implementation details
- Fixed-point inference end-to-end (Q8.8)
- Bit-cast from AXIS low 16 bits to fixed-point (`q88_from_axis`) to avoid float conversion hardware
- Loop pipelining and selective unrolling
- Weight array partitioning (gesture)
- `BIND_STORAGE` to BRAM for large weight ROM/buffers (voice)
- `bind_op` to DSP for multiply/add/sub (voice)
- Boundary-check removal via padded buffers (voice)
- On-the-fly temporal average pooling (voice) to reduce intermediate storage

---

## 3) Ultra96 Setup (Live + Video)

### Neural network implementation approach
- CNNs implemented in C++ using Vitis HLS and exported as IP.
- Vivado block design integrates two IPs (`gesture_cnn_0`, `voice_cnn_0`) plus AXI DMA(s) and AXI interconnect/control.

### Importing IP into Vivado
- HLS-generated IP is packaged and instantiated in Vivado design.
- Design is exported as overlay (`dual_cnn.xsa`) for PYNQ runtime loading.

### Input/output interfaces
- Input: AXI Stream (Q8.8 packed in `data[15:0]`)
- Output: AXI Stream single packet with predicted class id (`uint32`), `last=1`
- Control: AXI-Lite register interface (`ap_start`/`ap_done` style)

### Programming FPGA (board flow)
- Python script (`ultra96/dual_cnn_test.py`) loads overlay:
  - `overlay = Overlay('dual_cnn.xsa')`
- Resolves IP block names and DMA names dynamically.

### Low-level reads/writes handling
- Core control via MMIO-style writes:
  - `core.write(0x00, 0x01)` to start
  - `core.write(0x00, 0x00)` to stop
- DMA control and reset:
  - writes to DMA channel control register (`offset + 0x00`) for reset/run
- Data movement:
  - `sendchannel.transfer(in_buf)`, `recvchannel.transfer(out_buf)`, then `wait()`
- Buffer coherency:
  - input `flush()`, output `invalidate()`

---

## 4) Ultra96 Simulation Setup (Video)

### Tools used
- Vitis HLS C Simulation (C-Sim)
- Vitis HLS C/RTL Co-simulation (Co-Sim)

### Testbench structure
- Voice TB: `hls/voice_cnn_tb.cpp`
- Gesture TB: `hls/gesture_cnn_tb.cpp`
- Dataset vectors and expected predictions are compiled in:
  - `hls/voice_tb_cases.h`
  - `hls/gesture_tb_cases.h`
- Both testbenches evaluate 300 cases (`*_TB_NUM_CASES = 300`).

### Pass criteria implemented
- Accuracy gate: `>= 70%`
- Protocol correctness gate:
  - output class id within range
  - output `TLAST == 1`
  - exactly one output packet per inference

### Simulation result description
- Voice TB example run (from latest verified log):
  - `262/300` correct (`87.33%`), threshold `70%` -> PASS
  - protocol failures: `0`
  - C/RTL co-sim status: PASS
- Gesture TB uses the same gate structure and pass logic.

---

## 5) Software Implementation and Evaluation (Live + Video)

### Segmentation, features, and parameters

Gesture pipeline:
- Source windows filtered by `measurement_id` validity (`len == 60`, consistent label, full sequence).
- Features per timestep: `gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z`.
- Final model input shape: `[N, 6, 60]` for PyTorch Conv1d.

Voice pipeline:
- Audio resampled to `16 kHz`.
- MFCC extraction via `torchaudio.transforms.MFCC`:
  - `n_mfcc=40`, `n_fft=512`, `win_length=400`, `hop_length=160`, `n_mels=40`
- Time dimension normalized to `50` frames (crop/pad).
- Final model input shape: `[N, 40, 50]`.

### Training methodology
- Framework: PyTorch
- Loss: CrossEntropyLoss
- Optimizer: Adam
- LR scheduling: ReduceLROnPlateau
- Early stopping based on validation performance
- Reproducibility seeds used in notebooks

### Split strategy and validation
Both notebooks now follow the same policy:
1. Train/test split first
2. Train/validation split from train portion
3. Best checkpoint selected by validation accuracy
4. Final report on held-out test set

### Libraries/software used
- `torch`, `torchvision`, `torchaudio`
- `numpy`, `pandas`
- `scikit-learn` (`train_test_split`, `confusion_matrix`, metrics)
- `matplotlib`, `seaborn`
- PYNQ Python runtime on Ultra96 (`Overlay`, DMA, allocate)

### Confusion matrix and accuracy
- Notebooks produce two confusion matrices per model:
  - float evaluation
  - Q8.8 simulation evaluation
- Ultra96 end-to-end evidence (latest recorded in repo README context):
  - Gesture: `92.50%` on 120 samples
  - Voice: `75.00%` on 300 samples
- Voice software test (latest noted):
  - float: `77.28%`
  - Q8.8-sim: `77.28%`

---

## 6) Hardware Accelerator Implementation and Evaluation (Live + Video)

### Accuracy vs software implementation
- Hardware accuracy is close to software reference once these are aligned:
  - correct overlay/weights
  - correct feature order (`tc` for voice)
  - correct label mapping
  - fused-normalization deployment path consistency
- Observed trend:
  - Gesture hardware is stable and high.
  - Voice hardware now passes robust HLS dataset gate and approaches software performance.

### Execution time (inference + communication overhead)
`ultra96/dual_cnn_test.py` now records per-sample timing breakdown:
- `latency_total_*_ms` = prep + control + DMA path
- `latency_inference_*_ms` = control + DMA path
- `latency_comm_*_ms` = DMA submit + DMA wait
- `latency_prep_*_ms` = CPU preprocessing/packing overhead
- `latency_dma_submit_*_ms`
- `latency_dma_wait_*_ms`

This directly answers the requirement to include inference and communication overhead.

### Hardware resource usage
Gesture (`report/gesture_cnn_csynth.rpt`):
- BRAM: 33
- DSP: 70
- FF: 4,737
- LUT: 14,254
- Device utilization: BRAM 7%, DSP 19%, FF 3%, LUT 20%

Voice (latest tuned synthesis state discussed during implementation):
- DSP utilization reduced significantly from earlier over-limit state
- Latest target state achieved around:
  - LUT ~95%
  - DSP ~71%
- Further LUT reduction came from storage/mapping and arithmetic binding adjustments

### Optimization techniques implemented
- Fixed-point arithmetic (Q8.8)
- Saturation/truncation-aware quantized flow
- Input normalization fused into first conv weights/biases
- Loop pipelining / selected unrolling
- BRAM binding for weights/buffers
- DSP binding for arithmetic-heavy ops
- Removed boundary checks using padded feature buffers
- Reduced intermediate storage via on-the-fly pooling and compact output protocol

---

## 7) AI FPGA and Ultra96 - Power Management (Video)

### Implemented methods in current test flow
Power/performance control is integrated into `ultra96/dual_cnn_test.py` with runtime flags:

- `--cpu-governor <name>`
  - set Linux CPU governor (e.g., `performance`, `userspace`)
- `--cpu-freq-khz <freq>`
  - set CPU frequency in kHz (requires proper governor + privileges)
- `--pl-clock-mhz <mhz>`
  - set PL clock (`FCLK0`) via PYNQ Clocks API
- `--power-w <watts>`
  - provide measured board power manually
- `--power-sysfs-path <path>` and `--power-sysfs-scale <scale>`
  - read power from sensor path and scale to Watts

The script records before/after settings and warnings under `runtime_controls` in `summary.json`.

### Energy and overhead reporting
When power is available, the script computes energy estimates:
- `energy_total_mean_mj`
- `energy_inference_mean_mj`
- `energy_comm_mean_mj`

This ties power reporting to measured latency components.

### Disabling peripherals / switching off CPU cores
- Not automated inside `dual_cnn_test.py` yet (kept safe and non-destructive by default).
- Can be done via Linux runtime controls externally when needed for experiments.
- Typical method examples (manual, privileged):
  - disable unused services/peripherals via system service/network controls
  - offline selected CPU cores via `/sys/devices/system/cpu/cpuX/online`

### Practical power-management workflow used
1. Fix CPU governor/frequency.
2. Fix PL clock.
3. Run benchmark with fixed sample count.
4. Record latency breakdown + power + derived energy.
5. Compare across configurations for performance-per-Watt tradeoff.

---

## Suggested Demo Flow (for your assessment video)

1. Show notebook architecture and confusion matrices (float + Q8.8-sim).
2. Show HLS C-Sim/Cosim dataset gate with pass threshold.
3. Show Vivado/HLS utilization and timing.
4. Run `dual_cnn_test.py` on Ultra96 with fixed flags.
5. Show `summary.json` fields for accuracy, latency breakdown, runtime controls, and energy.
