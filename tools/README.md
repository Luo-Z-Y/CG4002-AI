# Tools

Helper scripts used around training, dataset preparation, export, and testbench generation.

This folder is for Python-side support code. The `hls/` folders should now stay focused on C++/header artefacts only.

## Files

### [gen_gesture_tb_cases.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/gen_gesture_tb_cases.py)

Builds [gesture_tb_cases.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/gesture/gesture_tb_cases.h) from `gesture_X_test.npy` and `gesture_y_test.npy`.

Purpose:

- sample a deterministic subset of gesture test cases
- pack them into Q8.8 AXIS words
- emit a C header for the gesture HLS testbench

### [gen_voice_tb_cases.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/gen_voice_tb_cases.py)

Builds [voice_tb_cases.h](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/hls/voice/voice_tb_cases.h) from `voice_X_test*.npy` and `voice_y_test.npy`.

Purpose:

- sample a deterministic subset of voice test cases
- pack them into Q8.8 AXIS words in time-major order
- emit a C header for the voice HLS testbench

Important note:

- for the current non-fused voice design, the correct HLS test input is the normalised test tensor, usually `voice_X_test_norm.npy`

### [voice_feature_pipeline.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/voice_feature_pipeline.py)

Shared deployment-aligned voice feature builder.

Purpose:

- decode `.wav` and `.m4a`
- run the same preprocessing path used by deployment
- build MFCC features for training
- scan multiple dated audio roots
- create source-aware speaker splits

This is the main reason voice training and deployment stay aligned.

### [preprocess_voice_audio.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/preprocess_voice_audio.py)

Offline waveform cleanup utility for listening checks and QA.

Purpose:

- trim silence
- normalise loudness
- export cleaned `.wav` files
- produce QA stats

This is not the active training pipeline for deployment parity.

### [import_gesture_packets.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/import_gesture_packets.py)

Appends raw gesture packet logs into an existing gesture CSV.

Purpose:

- parse packetised `.txt` logs
- assign measurement IDs
- extend the canonical training CSV format

Useful when you want to add more captured gesture files without editing the CSV manually.

### [merge_gesture_readings.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/merge_gesture_readings.py)

Merges multiple gesture packet log files into one combined ordered output.

Purpose:

- normalise class naming
- preserve packet boundaries
- produce a single merged collection file for review or later parsing

### [voice-gen.py](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/tools/voice-gen.py)

Legacy helper for bulk synthetic voice generation.

Purpose:

- call an external text-to-speech API
- generate many labelled audio clips for experiments

This is not part of the current deployment-aligned training path.

## Notebook Usage

The two training notebooks call the relevant helpers from this folder directly:

- [train_gesture_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_gesture_cnn.ipynb)
- [train_voice_cnn.ipynb](/Users/luozhiyang/Projects/CG4002-Code/CG4002-AI/notebooks/train_voice_cnn.ipynb)

After rerunning the notebooks on macOS, the repo-local `exports/` bundle is the intended handoff point for:

- Windows HLS work
- Ultra96 runtime files
- local Ultra96 test artefacts
