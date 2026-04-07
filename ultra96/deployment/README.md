# Ultra96 Deployment

Always-on MQTT inference bridge for the Ultra96 board.

This folder contains the runtime that:

- receives IMU and voice requests over MQTT
- preprocesses them on PS
- sends fixed-point tensors to the HLS accelerators over DMA
- maps the hardware argmax back to the configured label set
- publishes the result as a compact JSON packet

## Main Files

- [deployment.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/deployment.py): main bridge entry point
- [runtime.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/runtime.py): Ultra96 overlay and DMA wrapper
- [hardware.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/hardware.py): core names, DMA names, Q8.8 packing, label defaults
- [audio.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/audio.py): voice preprocessing and MFCC helpers
- [imu.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/imu.py): IMU trimming, baseline removal, and resampling
- [messages.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/messages.py): packet schema
- [dual_cnn.xsa](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/dual_cnn.xsa): deployed overlay

## Runtime Contract

### Gesture Path

1. Receive IMU MQTT payload.
2. Parse raw `gx, gy, gz, ax, ay, az` samples.
3. Enforce raw count band `15` to `300`.
4. Trim idle frames by gyro magnitude threshold `5.0`.
5. Enforce trimmed count band `40` to `300`.
6. Remove per-window baseline using the first `5` frames.
7. FFT-resample to `60 x 6`.
8. Apply software z-score normalisation using `gesture_mean.npy` and `gesture_std.npy`.
9. Pack to Q8.8 and send to the gesture HLS core.
10. Publish a gesture result packet.

Gesture uses non-fused weights. The software normalisation files must therefore match the exported gesture weights.

### Voice Path

1. Receive audio payload over MQTT.
2. Reconstruct chunked audio if required.
3. Decode to mono `16 kHz`.
4. Apply waveform preprocessing:
   - silence trim
   - loudness normalisation
5. Build MFCC `40 x 50`.
6. Apply software z-score normalisation using `voice_mean.npy` and `voice_std.npy`.
7. Pack to Q8.8 and send to the voice HLS core.
8. Publish a voice result packet.

Voice uses non-fused weights. The software normalisation files must therefore match the exported voice weights.

## Current Label Defaults

Defined in [hardware.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/hardware.py):

- Gesture: `Raise`, `Shake`, `Chop`, `Stir`, `Swing`, `Punch`
- Voice: `Bulbasaur`, `Charizard`, `Pikachu`

## MQTT Topics

Default topics from [common.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/common.py):

- subscribe IMU: `esp32/+/sensor/imu`
- subscribe voice: `phone/+/viz/mic`
- publish action: `ultra96/ai/action`
- publish pokemon: `ultra96/ai/pokemon`
- publish error: `ultra96/ai/error`

## Input JSON Formats

### IMU Request

The most canonical request format is:

```json
{
  "type": "imu",
  "data": {
    "samples": [
      {
        "gx": -12.5,
        "gy": 33.0,
        "gz": 8.0,
        "ax": 0.12,
        "ay": -0.34,
        "az": 9.81
      }
    ],
    "count": 60
  }
}
```

In practice, the live ESP32 topic may carry equivalent raw sample content that the bridge normalises into `ImuData`.

### Voice Request

The bridge accepts JSON carrying base64 audio bytes. A typical payload is:

```json
{
  "filename": "bulbasaur_01.m4a",
  "content_type": "audio/mp4",
  "audio_base64": "<base64 string>"
}
```

Chunked voice transport is also supported by the reconstruction layer.

## Output JSON Formats

### Gesture Result

Published to `ultra96/ai/action`:

```json
{
  "type": "action",
  "data": {
    "label": "Raise",
    "confidence": 1.0
  },
  "player": "1",
  "source_topic": "esp32/1/sensor/imu"
}
```

### Voice Result

Published to `ultra96/ai/pokemon`:

```json
{
  "type": "pokemon",
  "data": {
    "label": "Bulbasaur",
    "confidence": 1.0
  },
  "player": "1",
  "source_topic": "phone/1/viz/mic"
}
```

### Error Packet

Published to `ultra96/ai/error`:

```json
{
  "type": "error",
  "data": {
    "source_topic": "esp32/1/sensor/imu",
    "message": "error text here"
  },
  "player": "1"
}
```

Note: `confidence` is currently a placeholder default because the deployed hardware returns only the winning class index, not the full score vector.

## Required Files At Runtime

### Always required

- [dual_cnn.xsa](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/dual_cnn.xsa)

### Required for voice software normalisation

- [voice_mean.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_mean.npy)
- [voice_std.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/voice_std.npy)

### Required for gesture software normalisation

- [gesture_mean.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/gesture_mean.npy)
- [gesture_std.npy](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment/gesture_std.npy)

### Required in the overlay build itself

- the gesture and voice HLS IP must match the exported headers and Vivado design used to generate `dual_cnn.xsa`

## Example Run Commands

Default run:

```bash
cd /Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/ultra96/deployment
python3 deployment.py
```

Explicit voice normalisation paths:

```bash
python3 deployment.py \
  --voice-mean ./voice_mean.npy \
  --voice-std ./voice_std.npy
```

Explicit gesture normalisation paths:

```bash
python3 deployment.py \
  --gesture-mean ./gesture_mean.npy \
  --gesture-std ./gesture_std.npy
```

Explicit overlay, core names, and DMA names:

```bash
python3 deployment.py \
  --xsa-path dual_cnn.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0
```

Verbose debug mode:

```bash
python3 deployment.py --debug
```

## Captured Artefacts

Each runtime session creates a folder under:

- `deployment/captured/<session>/`

Useful subfolders:

- `raw_imu/`
- `resampled_imu/`
- `raw_audio/`
- `augmented_audio/`
- `mfcc/`

This is useful for checking what the board actually received and what was sent into the HLS core.

## Consistency Notes

### Gesture

- Training and deployment preprocessing are aligned.
- Gesture runtime expects non-fused weights plus software `gesture_mean/std`.

### Voice

- Training and deployment preprocessing are aligned through the shared deployment preprocessor.
- Voice runtime expects non-fused weights plus software `voice_mean/std`.

If you retrain either model, regenerate the relevant notebook artefacts before deploying again.
