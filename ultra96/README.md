# Ultra96

Ultra96-related runtime code, test utilities, and deployment support for the CG4002 AI system.

## Structure

```text
ultra96/
├── README.md
├── deployment/
├── local-ai-test/
└── mqtt-test/
```

## Subfolders

### deployment

The always-on MQTT inference bridge used on the Ultra96 board.

Key files:

- [deployment.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/deployment.py)
- [runtime.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/runtime.py)
- [audio.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/audio.py)
- [imu.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/imu.py)
- [hardware.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/hardware.py)

Detailed guide:

- [deployment/README.md](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/deployment/README.md)

### local-ai-test

Offline board-side testing against saved `.npy` datasets and the compiled overlay.

Key files:

- [dual_cnn_test.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/local-ai-test/dual_cnn_test.py)
- [run_assessment_suite.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/local-ai-test/run_assessment_suite.py)
- [dual_cnn.xsa](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/local-ai-test/dual_cnn.xsa)

Typical usage:

```bash
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/local-ai-test
python3 dual_cnn_test.py --mode both
```

### mqtt-test

MQTT smoke tests for the deployed bridge.

Key file:

- [self_test.py](/Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/mqtt-test/self_test.py)

Typical usage:

```bash
cd /Users/luozhiyang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Y4S2/CG4002/CG4002-Code/CG4002-AI/ultra96/mqtt-test
python3 self_test.py
```

## Current Runtime Design

### Gesture

- Input to Ultra96 runtime: preprocessed `60 x 6` IMU window
- Quantisation: Q8.8 packing on PS before DMA
- Normalisation: fused into exported gesture weights
- Output: winning class index from hardware, converted back to a gesture label in software

### Voice

- Input to Ultra96 runtime: normalised `40 x 50` MFCC
- Quantisation: Q8.8 packing on PS before DMA
- Normalisation: applied in software using `voice_mean.npy` and `voice_std.npy`
- Output: winning class index from hardware, converted back to a Pokemon label in software

## Topics And Results

The deployment bridge subscribes to:

- IMU: `esp32/+/sensor/imu`
- voice: `phone/+/viz/mic`

It publishes:

- gesture result: `ultra96/ai/action`
- voice result: `ultra96/ai/pokemon`
- errors: `ultra96/ai/error`

See the deployment README for the exact packet formats.

## Practical Recommendation

Use the local dashboard first to validate:

- gesture packet quality
- voice preprocessing quality
- model labels and confidence

Only move to Ultra96 deployment after the software-only dashboard path behaves correctly.
