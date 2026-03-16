# Ultra96

This folder is organized into three isolated subfolders.

## Structure

```text
ultra96/
├── README.md
├── local-ai-test/
├── mqtt-test/
└── deployment/
```

## local-ai-test

Purpose:

- run the local Ultra96 accelerator against saved gesture and voice test datasets

Contents:

- `dual_cnn_test.py`
- `run_assessment_suite.py`
- `dual_cnn.xsa`
- `gesture_X_test.npy`, `gesture_y_test.npy`
- `voice_X_test.npy`, `voice_y_test.npy`

Run from inside the folder:

```bash
cd local-ai-test
python3 dual_cnn_test.py --mode both
```

or

```bash
cd local-ai-test
python3 run_assessment_suite.py
```

## mqtt-test

Purpose:

- smoke-test the MQTT request/reply path

Contents:

- `self_test.py`
- `common.py`

Run from inside the folder after the deployment bridge is already running:

```bash
cd mqtt-test
python3 self_test.py
```

What it does:

- publishes one synthetic 60-sample IMU request to `esp32/1/sensor/imu`
- waits for one reply on `ultra96/ai/action`

## deployment

Purpose:

- run the always-on Ultra96 AI inference bridge for deployment

Contents:

- `deployment.py`
- `mqtt_ai_bridge.py`
- `common.py`
- `audio.py`
- `hardware.py`
- `messages.py`
- `runtime.py`
- `dual_cnn.xsa`

Run from inside the folder:

```bash
cd deployment
python3 deployment.py
```

Current default deployment settings:

- broker host: `13.238.81.254`
- broker port: `8883`
- CA file: `/home/xilinx/ca.crt`
- subscribe IMU: `esp32/+/sensor/imu`
- publish action: `ultra96/ai/action`
- publish pokemon: `ultra96/ai/pokemon`
- publish errors: `ultra96/ai/error`

Voice is disabled by default until `phone/<player>/viz/mic` is deployed.

## Notes

- each subfolder is intended to run on its own from within that folder
- `deployment/` is self-contained and does not depend on sibling folders
- `mqtt-test/` is only for smoke testing
- `local-ai-test/` is only for offline/local evaluation
