# Individual Assessment Answers (CG4002-AI)

This file answers all assessment prompts directly, based on the current implementation in this repository (`hls/`, `notebooks/`, `ultra96/`).

## 1) AI Model Design (Video)

### Model setup
Two separate 1D CNN models are implemented:

1. Gesture CNN (`notebooks/train_gesture_cnn.ipynb`, `hls/gesture_cnn.cpp`)
2. Voice CNN (`notebooks/train_voice_cnn.ipynb`, `hls/voice_cnn.cpp`)

Both are temporal models deployed as fixed-point HLS IP.

### Gesture CNN architecture
- Input tensor format: `[B, 6, 60]`
  - `B` = batch size
  - `6` channels = `gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z`
  - `60` timesteps per window
- Layer 1 block:
  - `Conv1d(6 -> 16, kernel_size=3, stride=1, padding=1)`
  - Shape: `[B, 6, 60] -> [B, 16, 60]`
  - Why `k=3`: captures short local temporal motion patterns while staying lightweight.
  - Why `padding=1`: preserves time length (`60`) so early feature alignment is not shifted.
  - `ReLU`: shape unchanged `[B, 16, 60]`
  - `MaxPool1d(kernel_size=2, stride=2)`: `[B, 16, 60] -> [B, 16, 30]`
  - Why max-pooling: downsamples temporal resolution, suppresses small noise spikes, and reduces compute.
- Layer 2 block:
  - `Conv1d(16 -> 32, kernel_size=3, stride=1, padding=1)`
  - Shape: `[B, 16, 30] -> [B, 32, 30]`
  - `ReLU`: shape unchanged `[B, 32, 30]`
  - `MaxPool1d(kernel_size=2, stride=2)`: `[B, 32, 30] -> [B, 32, 15]`
  - Why second pooling: further temporal compression and larger effective receptive field before dense layers.
- Flatten:
  - `[B, 32, 15] -> [B, 480]` (`32 x 15 = 480`)
- Classifier:
  - `Linear(480 -> 32)` + `ReLU`: `[B, 480] -> [B, 32]`
  - `Linear(32 -> 6)`: `[B, 32] -> [B, 6]` logits
- Output:
  - `Argmax` over 6 logits -> class ID per sample: `[B]`

Gesture neuron count (per sample, by stage):
- Input neurons: `6 x 60 = 360`
- Conv1 output neurons: `16 x 60 = 960`
- MaxPool1 output neurons: `16 x 30 = 480`
- Conv2 output neurons: `32 x 30 = 960`
- MaxPool2 output neurons: `32 x 15 = 480`
- FC1 neurons: `32`
- FC2/output neurons: `6`
- Final classification output: `1` class ID after argmax
- Total intermediate activations commonly reported (excluding argmax): `960 + 480 + 960 + 480 + 32 + 6 = 2918`

Gesture trainable parameter count:
- Conv1 params: `(16 x 6 x 3) + 16 = 288 + 16 = 304`
- Conv2 params: `(32 x 16 x 3) + 32 = 1536 + 32 = 1568`
- FC1 params: `(480 x 32) + 32 = 15360 + 32 = 15392`
- FC2 params: `(32 x 6) + 6 = 192 + 6 = 198`
- Total params: `304 + 1568 + 15392 + 198 = 17462`

Active gesture labels (public dataset mode):
- `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

### Voice CNN architecture
- Input tensor format: `[B, 40, 50]`
  - `40` MFCC coefficients over `50` time frames
- Layer 1 block:
  - `Conv1d(40 -> 16, kernel_size=3, stride=1, padding=1)`
  - Shape: `[B, 40, 50] -> [B, 16, 50]`
  - `ReLU`: shape unchanged `[B, 16, 50]`
  - `MaxPool1d(kernel_size=2, stride=2)`: `[B, 16, 50] -> [B, 16, 25]`
  - Why pooling here: removes frame-level redundancy and reduces downstream latency/cost.
- Layer 2 block:
  - `Conv1d(16 -> 32, kernel_size=3, stride=1, padding=1)`
  - Shape: `[B, 16, 25] -> [B, 32, 25]`
  - `ReLU`: shape unchanged `[B, 32, 25]`
  - `AdaptiveAvgPool1d(1)`: `[B, 32, 25] -> [B, 32, 1]`
  - Why adaptive average pooling: summarizes each channel over time into one value, avoids a large flatten+FC, and improves parameter efficiency for hardware.
- Classifier:
  - Squeeze time axis: `[B, 32, 1] -> [B, 32]`
  - `Linear(32 -> 3)`: `[B, 32] -> [B, 3]` logits
- Output:
  - `Argmax` over 3 logits -> class ID per sample: `[B]`

Voice neuron count (per sample, by stage):
- Input neurons: `40 x 50 = 2000`
- Conv1 output neurons: `16 x 50 = 800`
- MaxPool1 output neurons: `16 x 25 = 400`
- Conv2 output neurons: `32 x 25 = 800`
- AdaptiveAvgPool output neurons: `32 x 1 = 32`
- FC/output neurons: `3`
- Final classification output: `1` class ID after argmax
- Total intermediate activations commonly reported (excluding argmax): `800 + 400 + 800 + 32 + 3 = 2035`

Voice trainable parameter count:
- Conv1 params: `(16 x 40 x 3) + 16 = 1920 + 16 = 1936`
- Conv2 params: `(32 x 16 x 3) + 32 = 1536 + 32 = 1568`
- FC params: `(32 x 3) + 3 = 96 + 3 = 99`
- Total params: `1936 + 1568 + 99 = 3603`

Active voice labels:
- `marvin`, `sheila`, `visual`

### Public dataset provenance and references
- Gesture public data source:
  - UCI Machine Learning Repository, **Human Activity Recognition Using Smartphones** (6 activities include `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`)
  - Dataset page: `https://archive.ics.uci.edu/dataset/240`
  - DOI: `10.24432/C54S4K`
  - Repository note: current folder `data/gesture/27022026` is a compatibility/converted form for this project pipeline.
- Voice public data source:
  - Google/TensorFlow **Speech Commands** dataset (project uses selected classes: `marvin`, `sheila`, `visual`)
  - TFDS catalog page: `https://www.tensorflow.org/datasets/catalog/speech_commands`
  - Original v0.02 archive link used by the community/tooling: `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
  - Reference paper: Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv:1804.03209 (`https://arxiv.org/abs/1804.03209`)

### Activation functions used
- ReLU in hidden layers for both models
- No softmax in hardware output path (argmax on logits)

### Notes on normalization and quantization policy
- Training uses z-score normalization.
- Deployment folds z-score into the first convolution (`fuse_input_norm=True`) during weight export.
- Runtime board input is raw features; both IPs consume Q8.8 packed data.

---

## 2) FPGA Implementation (Video)

### Implementation overview (shared between gesture and voice)

Both accelerators follow the same top-level contract:
- Top functions:
  - `gesture_cnn(hls::stream<axis_t>&, hls::stream<axis_t>&)` in `hls/gesture_cnn.cpp:29`
  - `voice_cnn(hls::stream<axis_t>&, hls::stream<axis_t>&)` in `hls/voice_cnn.cpp:33`
- Interfaces:
  - AXIS in/out (`#pragma HLS INTERFACE axis`) and AXI-Lite control (`#pragma HLS INTERFACE s_axilite port=return`)
  - See `hls/gesture_cnn.cpp:30-32` and `hls/voice_cnn.cpp:34-36`
- Data representation:
  - internal tensor type `data_t = ap_fixed<16,8, AP_TRN, AP_SAT>` (Q8.8)
  - defined in `hls/gesture_typedefs.h:19` and `hls/voice_typedefs.h:24`
  - stream packet type `axis_t = ap_axiu<32,0,0,0>` (`gesture_typedefs.h:22`, `voice_typedefs.h:27`)
- Input packing contract:
  - low 16 bits of AXIS payload carry signed Q8.8 sample
  - converted by bit-cast helper `q88_from_axis(...)` (`gesture_cnn.cpp:21-27`, `voice_cnn.cpp:19-24`)
  - this avoids float-to-fixed conversion logic
- Output contract:
  - one AXIS packet carrying class ID in `data`
  - `keep = 0xF`, `strb = 0xF`, `last = 1`
  - implemented in `gesture_cnn.cpp:153-158` and `voice_cnn.cpp:191-196`

Core math used in both:
- Convolution: `y[o,t] = b[o] + sum_i sum_k x[i,t+k-p] * w[o,i,k]`
- ReLU: `max(0, x)`
- MaxPool1d(2): `max(x[o,2t], x[o,2t+1])`
- Fully connected: `z[c] = b[c] + sum_i h[i] * w[c,i]`
- Final class: `argmax(z)`

### Gesture CNN hardware mapping (`hls/gesture_cnn.cpp`)

Input and buffering:
- Input logical shape is `[6,60]`.
- Stream read loop is time-major then channel-major (`for t`, then `for c`) at `gesture_cnn.cpp:45-50`.
- Samples are stored into `input_buffer[c][t]` (`gesture_cnn.cpp:41-42,49`).
- `input_buffer` is partitioned by channel (`complete dim=1`) to increase parallel access (`gesture_cnn.cpp:42`).

Block 1 (`Conv1d 6->16, k=3, p=1` + ReLU + MaxPool2):
- Implemented as fused loops at `gesture_cnn.cpp:57-81`.
- Output buffer: `layer1[16][30]` (`gesture_cnn.cpp:54`).
- Per pooled time index `t` and subposition `p`:
  - `curr_t = t*2 + p` (`gesture_cnn.cpp:62`)
  - convolution sum starts from bias (`gesture_cnn.cpp:63`)
  - kernel loop `k=0..2` with channel loop `i=0..5` (`gesture_cnn.cpp:65-68`)
  - boundary protection via `if (in_t >= 0 && in_t < WINDOW_SIZE)` (`gesture_cnn.cpp:70`)
  - ReLU and local max selection (`gesture_cnn.cpp:77-79`)
- Effective shape flow: `[6,60] -> [16,60] -> [16,30]`.

Block 2 (`Conv1d 16->32, k=3, p=1` + ReLU + MaxPool2):
- Implemented at `gesture_cnn.cpp:88-113`.
- Output buffer: `layer2[32][15]` (`gesture_cnn.cpp:85`).
- Same fused conv+ReLU+pool pattern as block 1, with input from `layer1`.
- Shape flow: `[16,30] -> [32,30] -> [32,15]`.

FC stack:
- FC1 (`480 -> 32`) at `gesture_cnn.cpp:117-129`:
  - flatten index computed as `flat_idx = c*15 + t` (`gesture_cnn.cpp:123`)
  - weight index `w_idx = d*FLATTEN_SIZE + flat_idx` (`gesture_cnn.cpp:124`)
  - output `dense1[32]` with ReLU (`gesture_cnn.cpp:116,128`)
- FC2 (`32 -> 6`) at `gesture_cnn.cpp:132-141`, output `final_scores[6]`.
- Argmax at `gesture_cnn.cpp:143-151`.

Gesture-specific pragmas/optimization:
- Weight partitioning:
  - `conv1_w`, `conv2_w`, `fc1_w`, `fc2_w` partition pragmas at `gesture_cnn.cpp:35-38`
- Layer buffers partitioned by output channel (`layer1`, `layer2` complete dim=1 at `gesture_cnn.cpp:55,86`)
- Inner kernel loop unrolling (`gesture_cnn.cpp:68,99`)
- Pipeline pragmas on critical loops (`gesture_cnn.cpp:47,66,97,122,136`)
- Accumulator widening:
  - `conv_acc_t` and `fc_acc_t` in `gesture_cnn.cpp:13-14`

### Voice CNN hardware mapping (`hls/voice_cnn.cpp`)

Numeric and operator policy:
- Storage/IO still Q8.8 `data_t`, but accumulation uses `acc_t` and `mul_t` (`voice_cnn.cpp:8-10`).
- Multiplications are steered to DSP via `mul_dsp` + `bind_op` (`voice_cnn.cpp:27-30`).
- Additional DSP steering:
  - `bind_op op=add impl=dsp`
  - `bind_op op=sub impl=dsp`
  - see `voice_cnn.cpp:41-42`.

Memory placement policy:
- Weights forced into BRAM-backed ROM (`voice_cnn.cpp:49-54`).
- Feature buffers also BRAM-mapped (`voice_cnn.cpp:59-67`).
- This is a deliberate LUT-pressure reduction strategy.

Input and explicit padding:
- Input logical shape is `[40,50]`.
- Stream read order is also time-major then channel-major (`voice_cnn.cpp:80-86`).
- Data stored into `input_pad[c][t+1]` with left/right zeros (`voice_cnn.cpp:74-78,85`).
- This removes boundary branches for conv1.

Block 1 (`Conv1d 40->16, k=3, p=1` + ReLU + MaxPool2):
- Implemented at `voice_cnn.cpp:93-119`.
- Output buffer: `b1_out[16][25]` (`voice_cnn.cpp:62`).
- `curr_t = t*2 + p`, `pad_t = curr_t + 1` (`voice_cnn.cpp:100-101`).
- Convolution uses padded input directly: `input_pad[i][pad_t + (k - 1)]` (`voice_cnn.cpp:109`).
- Shape flow: `[40,50] -> [16,50] -> [16,25]`.

Intermediate padded buffer for block 2:
- `b1_pad[16][27]` built with zero borders (`voice_cnn.cpp:125-135`).
- This removes conv2 boundary checks as well.

Block 2 (`Conv1d 16->32, k=3, p=1` + ReLU + GlobalAvgPool):
- Implemented at `voice_cnn.cpp:142-163`.
- For each output channel `o`, loop over `t=0..24`:
  - compute conv + ReLU (`voice_cnn.cpp:149-160`)
  - accumulate temporal sum `sum_t` (`voice_cnn.cpp:144,160`)
- Global average is implemented as multiply by reciprocal:
  - `invT = 1/VOICE_B2_T` (`voice_cnn.cpp:142`)
  - `pooled[o] = sum_t * invT` (`voice_cnn.cpp:162`)
- This matches AdaptiveAvgPool1d(1) semantics from software.

FC and output:
- FC (`32 -> 3`) at `voice_cnn.cpp:170-177`, produces `logits[3]`.
- Argmax at `voice_cnn.cpp:182-189`.
- Output packet with `TLAST=1` at `voice_cnn.cpp:191-196`.

Voice-specific pragmas/optimization:
- `#pragma HLS ALLOCATION operation instances=mul limit=128` (`voice_cnn.cpp:39`)
- Pipeline II settings:
  - input/padding loops II=1 (`voice_cnn.cpp:75,83,126,132`)
  - block1 and block2 loops II=16 (`voice_cnn.cpp:95,146`)
  - FC inner loop II=2 (`voice_cnn.cpp:173`)
- On-the-fly temporal pooling avoids materializing full `[32,25]` conv2 output tensor.

### Shared structure vs differing structure

Shared:
- Same AXIS + AXI-Lite wrapper style, same Q8.8 external data format, same `q88_from_axis` unpacking, same argmax packet output protocol.
- Both fuse conv+ReLU+pooling behavior in loop nests for lower buffer overhead.

Different:
- Gesture keeps temporal information until flatten (`[32,15] -> FC1`), while voice compresses time via global average pooling (`[32,25] -> [32]`).
- Gesture leans on array partitioning + selective unroll; voice leans heavily on BRAM binding and DSP binding because its front-end channel count (`40`) makes conv1/conv2 much denser.
- Gesture uses explicit boundary checks in conv loops; voice uses padded buffers (`input_pad`, `b1_pad`) to remove boundary branches and reduce control/mux logic.

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

### Ultra96 command recipes (directly usable on board)
Assuming repository is cloned on Ultra96 and you run from repo root:

```bash
cd ~/CG4002-AI/ultra96
```

1. Gesture-only hardware test:
```bash
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode gesture \
  --gesture-features gesture_X_test.npy \
  --gesture-labels gesture_y_test.npy \
  --gesture-pack q88 \
  --save-dir ../report/evidence_dual \
  --tag gesture_baseline
```

2. Voice-only hardware test:
```bash
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode voice \
  --voice-features voice_X_test.npy \
  --voice-labels voice_y_test.npy \
  --voice-pack q88 \
  --voice-order tc \
  --save-dir ../report/evidence_dual \
  --tag voice_baseline
```

3. Combined run (both models):
```bash
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode both \
  --gesture-pack q88 \
  --voice-pack q88 \
  --voice-order tc \
  --save-dir ../report/evidence_dual \
  --tag both_baseline
```

Notes:
- If overlay block names differ, pass explicit names:
  - `--gesture-core <name> --voice-core <name> --gesture-dma <name> --voice-dma <name>`
- Script prints available IP names after loading overlay, which helps resolve naming mismatches.

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

I reviewed both training notebooks in detail:
- `notebooks/train_gesture_cnn.ipynb`
- `notebooks/train_voice_cnn.ipynb`

The software pipeline is implemented in PyTorch and structured to keep deployment consistency with the HLS path (Q8.8, optional normalization fusion into conv1).

### A) Gesture notebook: software implementation details

Data folder and run mode:
- The notebook automatically chooses the latest dated folder under `../data/gesture`.
- For the captured run, it used `../data/gesture/27022026`.
- Since `augmented_imudata.csv` already existed, TXT parsing/resampling/augmentation cells were skipped, and training used the prebuilt dataset directly.

Raw parser/resampler/augmenter logic (still implemented in notebook):
- TXT parsing supports `utf-16`, `utf-8-sig`, `utf-8`; strips null bytes; detects sensor blocks using `HEADER_MARKER = "YPR(deg)-X"`.
- Sensor rows are validated as 6 numeric values and mapped to:
  - `gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z`
- Resampling path standardizes each recording to `WINDOW_SIZE = 60`.
- Augmentation path includes jitter, scale, shift, optional temporal crop, then resample to 60.

Training-time window validation and split:
- A window is accepted only if:
  - exactly 60 rows
  - full unique `sequence_id` coverage
  - no NaN in sequence index
  - consistent `label_id` within the window
- Valid windows detected: `7352`.
- Leakage control: split by `measurement_id` first, not by row.
- Split strategy:
  - Train/Test = 80/20 (stratified)
  - Validation = 10% of train split (stratified)
- Saved split CSVs:
  - train rows: `317520`
  - val rows: `35340`
  - test rows: `88260`
- Window tensor shapes:
  - `X_train_raw = (5292, 60, 6)`
  - `X_val_raw = (589, 60, 6)`
  - `X_test_raw = (1471, 60, 6)`

Normalization and export-aligned artifacts:
- `StandardScaler` is fit on flattened train windows only.
- Saved deployment stats:
  - `mean.npy`: `[ 2.31284325e-03, -8.78569129e-04, -9.18387133e-04, -7.16640085e-04, -1.48521774e-05, -1.16569765e-04 ]`
  - `std.npy`: `[0.40777881, 0.38327034, 0.25755577, 0.19436369, 0.12311977, 0.10720748]`
- Raw board test arrays are also saved (`gesture_X_test.npy`, `gesture_y_test.npy`) so hardware can consume non-normalized input when normalization is fused into conv1.

Model/training configuration:
- Input to network: `[N, 6, 60]`.
- Architecture:
  - Conv1d(6->16, k=3, p=1) + ReLU + MaxPool1d(2)
  - Conv1d(16->32, k=3, p=1) + ReLU + MaxPool1d(2)
  - Flatten 480 -> FC(480->32) + ReLU -> FC(32->6)
- Hyperparameters:
  - batch size `16`, epochs `40`, Adam lr `1e-3`
  - `ReduceLROnPlateau(mode='max', factor=0.5, patience=5, min_lr=1e-5)`
  - early stopping patience `10`
- Training trace highlights:
  - Epoch 1: train `51.6%`, val `63.7%`
  - Epoch 20: train `92.6%`, val `89.3%`
  - Epoch 40: train `97.3%`, val `90.8%` (LR reduced to `0.0005`)
  - Best validation accuracy: `91.17%`

Gesture software evaluation:
- Final test accuracy (float): `93.27%`
- Final test accuracy (Q8.8 input simulation): `93.41%`
- Both float and Q8.8 confusion matrices are generated in notebook.
- Weight export uses `fuse_input_norm=True` and produces `gesture_cnn_weights.h` with `17462` parameters.

### B) Voice notebook: software implementation details

Data discovery and class mapping:
- The notebook automatically picks latest dated folder under `../data/audio`.
- For the captured run, it used `../data/audio/25022026`.
- Dataset scan found `5714` `.wav` files with class map:
  - `marvin -> 0`
  - `sheila -> 1`
  - `visual -> 2`
- Base class counts:
  - marvin: `2100`
  - sheila: `2022`
  - visual: `1592`

Feature extraction and augmentation:
- Audio preprocessing:
  - mono conversion
  - resample to `16 kHz`
- MFCC parameters:
  - `n_mfcc=40`, `n_fft=512`, `win_length=400`, `hop_length=160`, `n_mels=40`, `center=True`, `power=2.0`
- MFCC time axis fixed to `50` frames by crop/pad.
- Per-sample CMVN over time is applied before dataset-level normalization.
- Two augmentation implementations exist:
  - waveform-level (gain, shift, speed perturbation via resample, Gaussian noise)
  - feature-level (noise, time roll, time mask, frequency mask)

Split/caching/normalization behavior:
- Split policy is leakage-safe:
  - split at original sample index level first
  - then split validation from train
  - both stratified (`random_state=42`)
- In this run, `REUSE_SPLIT_IF_AVAILABLE=True` took fast path:
  - reused saved `voice_X_train.npy`, `voice_X_test_norm.npy`, `voice_y_train.npy`, `voice_y_test.npy`, mean/std.
  - rebuilt validation set from `X_base[val_idx]` normalized with saved train mean/std.
- Effective tensors used in training/eval:
  - `X_train=(12339, 40, 50)`
  - `X_val=(458, 40, 50)`
  - `X_test=(1143, 40, 50)`
- Raw board test tensor saved as `voice_X_test.npy` for fused-normalization deployment path.

Model/training configuration:
- Input to network: `[N, 40, 50]`.
- Active model (`VoiceCNN`):
  - Conv1d(40->16, k=3, p=1) + ReLU + MaxPool1d(2)
  - Conv1d(16->32, k=3, p=1) + ReLU + AdaptiveAvgPool1d(1)
  - FC(32->3)
- Hyperparameters:
  - batch size `16`, epochs `60`, Adam lr `1e-3`, weight decay `1e-4`
  - `ReduceLROnPlateau(mode='max', factor=0.5, patience=5, min_lr=1e-5)`
  - early stopping patience `10`
- Training trace highlights:
  - Epoch 1: train `74.3%`, val `83.8%`
  - Epoch 20: train `95.9%`, val `88.4%`
  - Epoch 25: train `97.6%`, val `88.2%` (LR dropped to `0.0005`)
  - Early stop at epoch `27`, best val `88.86%`

Voice software evaluation:
- Test accuracy (float): `85.74%`
- Test accuracy (Q8.8 input simulation): `85.74%`
- HLS-like layer-wise Q8.8 simulation: `85.83%`
- Deployment-equivalent simulation (raw board input + fused conv1 + layer-wise Q8.8): `86.00%`
- Confusion matrices are produced for:
  - float
  - Q8.8 input simulation
  - HLS-like fixed-point simulation
  - deployment-equivalent simulation
- Error analysis cell reports `163` misclassified samples (from `1143` test samples) and provides audio playback for manual inspection.
- Weight export uses `fuse_input_norm=True` and produces `voice_cnn_weights.h` with `3603` parameters.

### C) Software evaluation summary

Key software-side conclusions from notebook evidence:
- Split logic is explicitly leakage-aware in both notebooks (`measurement_id` split for gesture, source-sample split for voice).
- Quantization robustness is strong:
  - Gesture float vs Q8.8-sim gap is negligible (`93.27%` vs `93.41%`).
  - Voice float vs fixed-point proxies remains tightly aligned (`85.74%` to `86.00%` range across three simulations).
- Export flow is deployment-oriented:
  - train-time normalization statistics are persisted
  - normalization is fused into conv1 at export so Ultra96 can ingest raw features directly.

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

### Benchmark/evidence command set (Ultra96)
Single command to run assessment bundle (gesture + voice + both + power-profiled run):

```bash
cd ~/CG4002-AI/ultra96
python3 run_assessment_suite.py \
  --python-bin python3 \
  --runner dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --save-dir ../report/evidence_dual \
  --voice-order tc \
  --cpu-governor performance \
  --pl-clock-mhz 100 \
  --power-w 4.2 \
  --tag-prefix assess
```

Outputs:
- Per-run folder with `summary.json`
- Aggregated bundle:
  - `../report/evidence_dual/assessment_bundle_*.json`

### Hardware resource usage
Latest HLS C-synthesis estimates:

Gesture (`report/gesture_cnn_csynth.rpt`):
- BRAM_18K: `22` (`5%`)
- DSP: `8` (`2%`)
- FF: `3,437` (`2%`)
- LUT: `5,638` (`7%`)

Voice (`report/voice_cnn_csynth.rpt`):
- BRAM_18K: `27` (`6%`)
- DSP: `273` (`75%`)
- FF: `8,129` (`5%`)
- LUT: `26,636` (`37%`)

Combined HLS estimate (gesture + voice, naive sum):
- BRAM_18K: `49` (`~11%`)
- DSP: `281` (`~78%`)
- FF: `11,566` (`~8%`)
- LUT: `32,274` (`~46%`)

Latest Vivado post-implementation snapshot (`report/vivado-project-summary.png`):
- LUT: `34%`
- LUTRAM: `9%`
- FF: `15%`
- BRAM: `13%`
- DSP: `69%`
- BUFG: `1%`
- Timing: `WNS = +1.037 ns`, `TNS = 0 ns`, failing endpoints `0`
- DRC summary: `301 warnings`

Why Vivado can be lower than summed HLS estimates:
- HLS `csynth` numbers are per-IP estimates before full-chip logic optimization; summing them is conservative and can overstate final LUT/DSP.
- Vivado performs global optimization (`opt_design`/`phys_opt_design`) across the integrated netlist, trimming redundant logic and improving resource mapping.
- HLS reports include module-local control/interface overhead assumptions that do not always materialize one-to-one after integration.
- In this project, the main bottleneck resources (LUT and DSP) are lower in post-implementation (`34%`, `69%`) than naive combined HLS (`~46%`, `~78%`), while FF/BRAM can still rise due to system-level glue logic (DMA, AXI interconnect, buffering, control paths).

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

### Runtime control command examples (Ultra96)
1. Fix CPU governor to performance:
```bash
cd ~/CG4002-AI/ultra96
python3 dual_cnn_test.py --xsa-path dual_cnn.xsa --mode both --cpu-governor performance --tag gov_performance
```

2. Set userspace governor + CPU frequency (example: 1.2 GHz):
```bash
cd ~/CG4002-AI/ultra96
sudo python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode both \
  --cpu-governor userspace \
  --cpu-freq-khz 1200000 \
  --tag cpu_1200mhz
```

3. Set PL clock (FCLK0):
```bash
cd ~/CG4002-AI/ultra96
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode both \
  --pl-clock-mhz 100 \
  --tag pl100
```

4. Manual power entry for energy estimation:
```bash
cd ~/CG4002-AI/ultra96
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode both \
  --power-w 4.2 \
  --tag power_manual
```

5. Read power from sysfs sensor (example path):
```bash
cd ~/CG4002-AI/ultra96
python3 dual_cnn_test.py \
  --xsa-path dual_cnn.xsa \
  --mode both \
  --power-sysfs-path /sys/class/hwmon/hwmon0/power1_input \
  --power-sysfs-scale 1e-6 \
  --tag power_sysfs
```

Practical note:
- Frequency/governor writes may require root privileges; if unavailable, script continues and records warnings in `runtime_controls.warnings` inside `summary.json`.

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
