# CG4002-AI

AI training, local testing, HLS export, and Ultra96 deployment support for the CG4002 project.

This workspace currently supports:

- gesture classification with `6` classes: `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`
- voice classification with `3` classes: `Bulbasaur`, `Charizard`, `Pikachu`
- local browser-based testing for both models
- HLS header export for the Windows Vitis flow
- Ultra96 MQTT deployment with separate gesture and voice accelerators

## Current Source Of Truth

### Gesture

- Training notebook: [train_gesture_cnn.ipynb](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/notebooks/train_gesture_cnn.ipynb)
- Active retraining dataset override: [data/gesture/20260328peer](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/gesture/20260328peer)
- Dataset content: peer-only raw gyro/acc gesture packets, `50` measurements per class, `300` measurements total
- Preprocessing contract:
  - raw count band: `15` to `300`
  - motion trim by gyro magnitude threshold `5.0`
  - trimmed count band: `40` to `300`
  - baseline removal from first `5` frames
  - FFT resample to `60 x 6`
- Runtime normalisation strategy: fused into exported gesture `conv1`

### Voice

- Training notebook: [train_voice_cnn.ipynb](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/notebooks/train_voice_cnn.ipynb)
- Active combined sources:
  - [data/audio/20260313](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/audio/20260313)
  - [data/audio/20260321](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/audio/20260321)
- Artefact folder: [data/audio/combined](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/audio/combined)
- Test policy: hold out `5` speakers from `20260321`
- Preprocessing contract:
  - decode raw audio to mono `16 kHz`
  - silence trim
  - loudness normalisation
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
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI
./.venv/bin/python --version
```

If shell activation is unreliable on your machine, prefer:

```bash
./.venv/bin/python <script.py>
```

## Training Workflow

### Gesture Retraining

Run the notebook top to bottom:

```bash
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/notebooks
jupyter notebook
```

Then open [train_gesture_cnn.ipynb](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/notebooks/train_gesture_cnn.ipynb).

Important notes:

- The notebook is currently pinned to [data/gesture/20260328peer](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/gesture/20260328peer) via `GESTURE_DIR_OVERRIDE`.
- This is a peer-only dataset with balanced classes.
- Gesture export still uses `fuse_input_norm=True`.

Main gesture artefacts produced under the selected gesture folder:

- `gesture_X_test.npy`
- `gesture_y_test.npy`
- `mean.npy`
- `std.npy`
- `preprocess_meta.json`
- `gesture_cnn_weights.h`

### Voice Retraining

Run [train_voice_cnn.ipynb](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/notebooks/train_voice_cnn.ipynb) top to bottom.

Important notes:

- The notebook uses the shared deployment-aligned pipeline in [voice_feature_pipeline.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/tools/voice_feature_pipeline.py).
- `USE_CLEANED_AUDIO` should stay `False` for deployment parity.
- The active output label is `combined`.
- The current split holds out `5` speakers from `20260321` for test.
- Voice export uses `fuse_input_norm=False`.

Main voice artefacts produced in [data/audio/combined](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/audio/combined):

- `voice_manifest.csv`
- `voice_features.npy`
- `voice_X_train.npy`
- `voice_X_test.npy`
- `voice_X_test_norm.npy`
- `voice_y_train.npy`
- `voice_y_test.npy`
- `voice_mean.npy`
- `voice_std.npy`
- `voice_cnn_weights.h`

## HLS Export And Windows Flow

After fully rerunning the notebooks, the usual files to copy to your Windows HLS workspace are:

### Gesture

- [data/gesture/20260328peer/gesture_cnn_weights.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/gesture/20260328peer/gesture_cnn_weights.h)
- [hls/gesture/gesture_cnn.cpp](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/gesture/gesture_cnn.cpp)
- [hls/gesture/gesture_cnn.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/gesture/gesture_cnn.h)
- [hls/gesture/gesture_typedefs.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/gesture/gesture_typedefs.h)

### Voice

- [data/audio/combined/voice_cnn_weights.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/data/audio/combined/voice_cnn_weights.h)
- [hls/voice/voice_cnn.cpp](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/voice/voice_cnn.cpp)
- [hls/voice/voice_cnn.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/voice/voice_cnn.h)
- [hls/voice/voice_typedefs.h](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/hls/voice/voice_typedefs.h)

For voice deployment, the software path also needs:

- [ultra96/deployment/voice_mean.npy](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/voice_mean.npy)
- [ultra96/deployment/voice_std.npy](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/voice_std.npy)

## Local Dashboard

The local dashboard is useful for testing models before deploying to Ultra96.

Start it with:

```bash
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI
./.venv/bin/python dashboard/server.py --host 127.0.0.1 --port 8000
```

Optional explicit artefact paths:

```bash
./.venv/bin/python dashboard/server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --gesture-weights data/gesture/20260328peer/gesture_cnn_weights.h \
  --voice-weights data/audio/combined/voice_cnn_weights.h \
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

## Live Gesture Testing From Hardware

Use the hardware bridge script in the hardware workspace:

- [gesture_live_test.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-HW/gesture_live_test.py)

Example:

```bash
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-HW
./.venv/bin/python gesture_live_test.py \
  --port /dev/cu.usbserial-1410 \
  --baud 115200 \
  --server-url http://127.0.0.1:8000/api/gesture/infer
```

## Consistency Summary

### Gesture

- Training preprocessing and deployment preprocessing are aligned.
- Dashboard inference uses the same preprocessing contract.
- Gesture normalisation is fused into exported weights.
- Ultra96 and dashboard both expect the same class order:
  - `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`

### Voice

- Training feature generation uses the deployment preprocessor.
- Dashboard and Ultra96 both apply `voice_mean/std` in software.
- Voice export is intentionally non-fused.
- Ultra96 and dashboard both expect the same class order:
  - `Bulbasaur`, `Charizard`, `Pikachu`

## Ultra96 Deployment

See:

- [ultra96/README.md](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/README.md)
- [ultra96/deployment/README.md](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/README.md)

The deployment README documents:

- MQTT topics
- input and output JSON packets
- capture directories
- required runtime files
- example run commands
