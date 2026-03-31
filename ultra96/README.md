# Ultra96

This folder contains the board-side runtime, offline board tests, and MQTT smoke tests for the CG4002 AI system.

The key point is that the commands below are written from the `ultra96/` folder itself. That is the layout you should keep when you copy the runtime to the Ultra96 board.

## Folder layout

```text
ultra96/
├── README.md
├── deployment/
│   ├── README.md
│   ├── deployment.py
│   ├── runtime.py
│   ├── hardware.py
│   ├── audio.py
│   ├── imu.py
│   ├── messages.py
│   ├── common.py
│   ├── reconstruct.py
│   ├── mqtt_ai_bridge.py
│   ├── dual_cnn.xsa
│   ├── voice_mean.npy
│   └── voice_std.npy
├── local-ai-test/
│   ├── dual_cnn_test.py
│   ├── run_assessment_suite.py
│   ├── dual_cnn.xsa
│   ├── gesture_X_test.npy
│   ├── gesture_y_test.npy
│   ├── voice_X_test.npy
│   ├── voice_X_test_raw.npy
│   └── voice_y_test.npy
└── mqtt-test/
    ├── common.py
    └── self_test.py
```

## What each subfolder is for

### `deployment/`

The always-on MQTT inference bridge that runs on the Ultra96 board.

Use this when you want the board to:

- subscribe to IMU packets
- subscribe to voice packets
- preprocess data on PS
- send tensors to the HLS accelerators over DMA
- publish final action and Pokemon labels back over MQTT

Important runtime files:

- `deployment.py`: main CLI entry point
- `runtime.py`: overlay loading and DMA execution
- `hardware.py`: core names, DMA names, Q8.8 helpers, label defaults
- `audio.py`: voice preprocessing and software MFCC normalisation
- `imu.py`: gesture trimming, baseline removal, and resampling
- `dual_cnn.xsa`: board overlay
- `voice_mean.npy`, `voice_std.npy`: voice software normalisation stats

### `local-ai-test/`

Offline board-side testing against saved `.npy` test tensors.

Use this when you want to validate the hardware path on Ultra96 without MQTT, browsers, or live devices.

Important files:

- `dual_cnn_test.py`: main offline hardware test runner
- `run_assessment_suite.py`: convenience wrapper for repeatable assessments
- `dual_cnn.xsa`: overlay used by the local board test
- `gesture_X_test.npy`, `gesture_y_test.npy`: gesture test cases
- `voice_X_test.npy`, `voice_y_test.npy`: voice test cases used by current non-fused voice HLS flow
- `voice_X_test_raw.npy`: raw MFCC reference for debugging only

### `mqtt-test/`

Simple MQTT smoke tests for the deployed bridge.

Use this when the deployment bridge is already running and you want to confirm that MQTT request and reply flow works.

Important files:

- `self_test.py`: publishes a synthetic request and waits for the Ultra96 reply
- `common.py`: shared MQTT defaults for the smoke test

## Current runtime contract

### Gesture

- Input to hardware: preprocessed `60 x 6` IMU window
- Preprocessing: trim idle frames, baseline removal, FFT resample
- Quantisation: Q8.8 on PS before DMA
- Normalisation: fused into the exported gesture weights

### Voice

- Input to hardware: software-normalised `40 x 50` MFCC
- Preprocessing: decode, trim silence, loudness normalise, MFCC extraction
- Quantisation: Q8.8 on PS before DMA
- Normalisation: applied in software using `deployment/voice_mean.npy` and `deployment/voice_std.npy`

## CLI usage

### 1. Run the live deployment bridge

From the `ultra96/` folder:

```bash
cd ultra96
python3 deployment/deployment.py
```

Useful variants:

```bash
python3 deployment/deployment.py \
  --xsa-path deployment/dual_cnn.xsa \
  --voice-mean deployment/voice_mean.npy \
  --voice-std deployment/voice_std.npy
```

```bash
python3 deployment/deployment.py \
  --host <broker-host> \
  --username <mqtt-username> \
  --password <mqtt-password>
```

For the full bridge details, packet formats, and all flags, see `deployment/README.md`.

### 2. Run offline board tests

From the `ultra96/` folder:

```bash
cd ultra96
python3 local-ai-test/dual_cnn_test.py --mode both
```

Common variants:

```bash
python3 local-ai-test/dual_cnn_test.py --mode gesture
```

```bash
python3 local-ai-test/dual_cnn_test.py --mode voice
```

```bash
python3 local-ai-test/dual_cnn_test.py \
  --xsa-path local-ai-test/dual_cnn.xsa \
  --gesture-features local-ai-test/gesture_X_test.npy \
  --gesture-labels local-ai-test/gesture_y_test.npy \
  --voice-features local-ai-test/voice_X_test.npy \
  --voice-labels local-ai-test/voice_y_test.npy
```

### 3. Run MQTT smoke tests

From the `ultra96/` folder:

```bash
cd ultra96
python3 mqtt-test/self_test.py
```

Useful variant:

```bash
python3 mqtt-test/self_test.py \
  --host <broker-host> \
  --username <mqtt-username> \
  --password <mqtt-password>
```

## Practical workflow

1. Run the training notebooks on your Mac.
2. Copy the refreshed `ultra96/` runtime bundle to the board.
3. Replace `deployment/dual_cnn.xsa` and `local-ai-test/dual_cnn.xsa` with the hardware build you generated on Windows.
4. Run `local-ai-test/dual_cnn_test.py` first.
5. When offline tests are correct, run `deployment/deployment.py`.
6. Use `mqtt-test/self_test.py` or the real devices to validate end-to-end behaviour.

## Notes

- `deployment/voice_mean.npy` and `deployment/voice_std.npy` are required for the current voice runtime.
- Gesture does not need separate runtime `mean/std` files because gesture normalisation is fused into the exported weights.
- The overlay path examples above are written relative to `ultra96/`, not to your Mac workspace root.
