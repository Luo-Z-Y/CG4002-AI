# CG4002-AI

AI training, local testing, HLS export, and Ultra96 deployment support for the CG4002 project.

This workspace currently supports:

- gesture classification with `6` classes: `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`
- voice classification with `6` classes: `Bulbasaur`, `Charizard`, `Greninja`, `Lugia`, `Mewtwo`, `Pikachu`
- local browser-based testing for both models
- HLS header export for the Windows Vitis flow
- Ultra96 MQTT deployment with separate gesture and voice accelerators

## Current Source Of Truth

### Gesture

- Training notebook: [train_gesture_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_gesture_cnn.ipynb)
- Current dated dataset folder: [data/gesture/20260406](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/gesture/20260406)
- Dataset content: merged raw gyro/acc gesture packets, `100` measurements per class, `600` measurements total
- Preprocessing contract:
  - raw count band: `15` to `300`
  - motion trim by gyro magnitude threshold `5.0`
  - trimmed count band: `40` to `300`
  - baseline removal from first `5` frames
  - FFT resample to `60 x 6`
- Runtime normalisation strategy: software z-score using `gesture_mean.npy` and `gesture_std.npy`

### Voice

- Training notebook: [train_voice_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_voice_cnn.ipynb)
- Active combined artefact folder: [data/audio/20260406_combined](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/audio/20260406_combined)
- Active raw source buckets:
  - [data/audio/20260406/new](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/audio/20260406/new)
  - [data/audio/20260406/old](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/audio/20260406/old)
  - [data/audio/20260406/synthetic](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/audio/20260406/synthetic)
  - reviewed dashboard voice clips under [dashboard/data](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/dashboard/data)
- Current runtime normalisation files: [ultra96/deployment/voice_mean.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_mean.npy) and [ultra96/deployment/voice_std.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_std.npy)
- Test policy: stratified random file split, randomised per run by default unless you pin `SPLIT_SEED`
- Preprocessing contract:
  - decode raw audio to mono `16 kHz`
  - tighter silence trim plus focus windowing
  - pre-emphasis
  - cepstral mean normalisation before dataset z-score
  - MFCC extraction to `40 x 50`
- Runtime normalisation strategy: software MFCC z-score using `voice_mean.npy` and `voice_std.npy`

## Repository Layout

```text
CG4002-AI/
├── README.md
├── dashboard/
│   ├── server.py
│   ├── runtime.py
│   └── static/
├── data/
│   ├── audio/
│   └── gesture/
├── hls/
│   ├── gesture/
│   └── voice/
├── notebooks/
│   ├── train_gesture_cnn.ipynb
│   └── train_voice_cnn.ipynb
├── tools/
│   ├── import_gesture_packets.py
│   ├── preprocess_voice_audio.py
│   └── voice_feature_pipeline.py
└── ultra96/
    ├── README.md
    ├── deployment/
    ├── local-ai-test/
    └── mqtt-test/
```

## Local Environment

The recommended way to run anything in this workspace is to use the project venv directly:

```bash
cd /Users/luozhiyang/Projects/CG4002-Code/CG4002-AI
./.venv/bin/python --version
```

If shell activation is unreliable on your machine, prefer the project venv directly:

```bash
./.venv/bin/python <script.py>
```

## Training Workflow

### Gesture Retraining

Run the notebook top to bottom:

```bash
cd /Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks
jupyter notebook
```

Then open [train_gesture_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_gesture_cnn.ipynb).

Important notes:

- The notebook now auto-selects the latest dated gesture folder when `GESTURE_DIR_OVERRIDE = None`.
- The current latest folder is [data/gesture/20260406](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/gesture/20260406).
- `imudata.csv` is auto-built from flat txt logs if needed.
- Gesture export uses `fuse_input_norm=False`.

Main gesture artefacts produced under the selected gesture folder:

- `gesture_X_test.npy`
- `gesture_y_test.npy`
- `mean.npy`
- `std.npy`
- `preprocess_meta.json`
- `gesture_cnn_weights.h`

### Voice Retraining

Run [train_voice_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_voice_cnn.ipynb) top to bottom.

Important notes:

- The notebook uses the shared deployment-aligned pipeline in [voice_feature_pipeline.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/voice_feature_pipeline.py).
- `USE_CLEANED_AUDIO` should stay `False` for deployment parity.
- The active output label is `20260406_combined`.
- The notebook currently supports two voice model variants:
  - `deployed`: current 16/32-channel HLS-compatible model
  - `experimental`: modestly wider 20/40-channel notebook-only model
- Voice export uses `fuse_input_norm=False`.
- The dashboard can load a notebook checkpoint directly from `voice_dashboard_model.pt`, so local dashboard testing is no longer limited to the current HLS-compatible voice shape.

Main voice artefacts are expected under the selected audio artefact folder:

- `voice_manifest.csv`
- `voice_features.npy`
- `voice_X_train.npy`
- `voice_X_test.npy`
- `voice_X_test_norm.npy`
- `voice_y_train.npy`
- `voice_y_test.npy`
- `voice_mean.npy`
- `voice_std.npy`
- `voice_labels.json`
- `voice_preprocess_config.json`
- `voice_test_predictions.csv`
- `voice_dashboard_model.pt`
- `voice_cnn_weights.h`

## HLS Export And Windows Flow

After fully rerunning the notebooks, the usual files to copy to your Windows HLS workspace are:

### Gesture

- [data/gesture/20260406/gesture_cnn_weights.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/data/gesture/20260406/gesture_cnn_weights.h)
- [hls/gesture/gesture_cnn.cpp](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/gesture/gesture_cnn.cpp)
- [hls/gesture/gesture_cnn.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/gesture/gesture_cnn.h)
- [hls/gesture/gesture_typedefs.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/gesture/gesture_typedefs.h)

### Voice

- [hls/voice/voice_cnn_weights.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_cnn_weights.h)
- [hls/voice/voice_cnn.cpp](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_cnn.cpp)
- [hls/voice/voice_cnn.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_cnn.h)
- [hls/voice/voice_typedefs.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_typedefs.h)

Only the `deployed` voice variant exports to the current HLS header layout. The `experimental` voice variant stays notebook and dashboard only until the HLS voice IP is widened to match it.

For voice deployment, the software path also needs:

- [ultra96/deployment/voice_mean.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_mean.npy)
- [ultra96/deployment/voice_std.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_std.npy)

For gesture deployment, the software path also needs:

- [ultra96/deployment/gesture_mean.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/gesture_mean.npy)
- [ultra96/deployment/gesture_std.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/gesture_std.npy)

## Local Dashboard

The local dashboard is useful for testing models before deploying to Ultra96.

Start it with:

```bash
cd /Users/luozhiyang/Projects/CG4002-Code/CG4002-AI
./.venv/bin/python dashboard/server.py --host 127.0.0.1 --port 8000
```

Optional explicit artefact paths:

```bash
  ./.venv/bin/python dashboard/server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --gesture-weights data/gesture/20260406/gesture_cnn_weights.h \
  --gesture-mean ultra96/deployment/gesture_mean.npy \
  --gesture-std ultra96/deployment/gesture_std.npy \
  --voice-weights hls/voice/voice_cnn_weights.h \
  --voice-mean ultra96/deployment/voice_mean.npy \
  --voice-std ultra96/deployment/voice_std.npy
```

The dashboard supports:

- live gesture inference pushed from the ESP32 bridge
- voice file upload
- browser microphone recording
- raw and processed IMU plots
- raw and processed waveform plots
- processed waveform playback
- processed MFCC visualisation
- exact IMU tables
- a `Cleanup` tab that reads `voice_test_predictions.csv` and lets you play or delete misclassified test clips

Voice model loading in the dashboard now works like this:

- prefer the newest `voice_dashboard_model.pt` checkpoint if one exists
- otherwise fall back to [hls/voice/voice_cnn_weights.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_cnn_weights.h)

That means the dashboard can test the current `experimental` notebook model even when Ultra96 and HLS still use the older deployed voice shape.

## Live Gesture Testing From Hardware

Use the hardware bridge script in the hardware workspace:

- [gesture_live_test.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-HW/gesture_live_test.py)

Example:

```bash
cd /Users/luozhiyang/Projects/CG4002-Code/CG4002-HW
./.venv/bin/python gesture_live_test.py \
  --port /dev/cu.usbserial-1410 \
  --baud 115200 \
  --server-url http://127.0.0.1:8000/api/gesture/infer
```

## Consistency Summary

### Gesture

- Training preprocessing and deployment preprocessing are aligned.
- Dashboard inference uses the same preprocessing contract.
- Dashboard and Ultra96 both apply `gesture_mean/std` in software.
- Gesture export is intentionally non-fused.
- Ultra96 and dashboard both expect the same class order:
  - `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`

### Voice

- Training feature generation uses the deployment preprocessor.
- Dashboard and Ultra96 both apply `voice_mean/std` in software.
- Voice export is intentionally non-fused for the deployed HLS path.
- The dashboard may instead load `voice_dashboard_model.pt` for notebook-parity local testing.
- Ultra96 and dashboard both expect the same class order:
  - `Bulbasaur`, `Charizard`, `Greninja`, `Lugia`, `Mewtwo`, `Pikachu`

## Ultra96 Deployment

See:

- [ultra96/README.md](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/README.md)
- [ultra96/deployment/README.md](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/README.md)

The deployment README documents:

- MQTT topics
- input and output JSON packets
- capture directories
- required runtime files
- example run commands
