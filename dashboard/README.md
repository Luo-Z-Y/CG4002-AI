# Dashboard

This folder contains the local browser dashboard used to test gesture and voice inference without deploying to the Ultra96 board first.

Use it when you want:

- browser voice recording
- audio file upload
- live gesture packets forwarded from the ESP32 helper
- raw and processed data visualisation
- model predictions with labels and confidence

## Folder layout

```text
dashboard/
├── README.md
├── server.py
├── runtime.py
└── static/
    ├── index.html
    ├── app.js
    └── styles.css
```

## What each file does

### `server.py`

Starts the local HTTP server and exposes the dashboard API endpoints.

Main responsibilities:

- serve the web page
- receive voice uploads and browser recordings
- receive gesture packets from the hardware helper
- load the requested weights and voice normalisation files
- keep the latest dashboard state

### `runtime.py`

Runs the actual local inference pipeline.

Main responsibilities:

- gesture preprocessing and inference
- voice preprocessing and inference
- loading exported HLS headers into the local Python runtime
- applying software voice `mean/std`
- preparing waveform, MFCC, and IMU data for visualisation

### `static/index.html`

The dashboard page structure.

### `static/app.js`

Browser-side logic.

Main responsibilities:

- call the dashboard API
- handle file upload and microphone recording
- poll for the latest state
- update prediction cards, plots, and tables
- switch between the gesture and voice views

### `static/styles.css`

Dashboard styling and layout rules.

## CLI usage

Run the dashboard from the `CG4002-AI/` folder:

```bash
cd CG4002-AI
./.venv/bin/python dashboard/server.py --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

### Useful flags

Use custom exported weights:

```bash
./.venv/bin/python dashboard/server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --gesture-weights /absolute/path/to/gesture_cnn_weights.h \
  --voice-weights /absolute/path/to/voice_cnn_weights.h
```

Use custom voice normalisation files:

```bash
./.venv/bin/python dashboard/server.py \
  --voice-mean /absolute/path/to/voice_mean.npy \
  --voice-std /absolute/path/to/voice_std.npy
```

You only need `--voice-mean` and `--voice-std` when you want to override the defaults. The current voice path expects non-fused weights plus software normalisation.

## Gesture testing flow

1. Start the dashboard server.
2. In `CG4002-HW`, run the ESP32 bridge:

```bash
./.venv/bin/python gesture_live_test.py \
  --port /dev/cu.<your-device> \
  --baud 115200 \
  --server-url http://127.0.0.1:8000/api/gesture/infer
```

3. Perform a gesture.
4. Check the dashboard for:
   - predicted label
   - confidence
   - raw IMU plot
   - processed IMU plot
   - exact numeric IMU tables

## Voice testing flow

1. Start the dashboard server.
2. Open the page in the browser.
3. Either upload a file or record with the browser microphone.
4. Check the dashboard for:
   - prediction and confidence
   - original playback
   - processed playback
   - raw waveform
   - processed waveform
   - processed MFCC

## Current model assumptions

### Gesture

- dashboard gesture preprocessing matches deployment preprocessing
- gesture input normalisation is fused into the exported gesture weights

### Voice

- dashboard voice preprocessing matches deployment preprocessing
- dashboard applies software `voice_mean.npy` and `voice_std.npy`
- voice weights should therefore be exported non-fused

## Practical recommendation

Use the dashboard before testing on Ultra96. It is faster to inspect:

- bad gesture packet segmentation
- voice preprocessing issues
- wrong labels
- dataset drift between training samples and live samples

Once the dashboard behaves correctly, move the same exported artefacts into the Ultra96 runtime.
