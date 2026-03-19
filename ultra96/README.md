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
- `imu.py`
- `hardware.py`
- `messages.py`
- `runtime.py`
- `dual_cnn.xsa`

Run from inside the folder:

```bash
cd deployment
python3 deployment.py
```

Example CLI commands:

1. Default deployment bridge:

```bash
cd deployment
python3 deployment.py
```

2. Explicit broker, credentials, and TLS CA:

```bash
cd deployment
python3 deployment.py \
  --host 13.238.81.254 \
  --port 8883 \
  --username mqttuser \
  --password cg4002 \
  --cafile /home/xilinx/ca.crt
```

3. Same as above but with a custom client ID and longer connect timeout:

```bash
cd deployment
python3 deployment.py \
  --client-id ultra96-ai-board1 \
  --connect-timeout-s 20 \
  --keepalive 60
```

4. Enable voice subscription with the intended deployment topic:

```bash
cd deployment
python3 deployment.py \
  --voice-topic 'phone/+/viz/mic'
```

5. Override all MQTT topics explicitly:

```bash
cd deployment
python3 deployment.py \
  --imu-topic 'esp32/+/sensor/imu' \
  --voice-topic 'phone/+/viz/mic' \
  --action-topic 'ultra96/ai/action' \
  --pokemon-topic 'ultra96/ai/pokemon' \
  --error-topic 'ultra96/ai/error'
```

6. Run with custom overlay path and explicit IP/DMA names:

```bash
cd deployment
python3 deployment.py \
  --xsa-path dual_cnn.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0
```

7. Tune runtime timeout and placeholder confidence:

```bash
cd deployment
python3 deployment.py \
  --timeout-s 2.0 \
  --default-confidence 1.0
```

8. Accept variable-length IMU windows and resample them to the 60-step model input:

```bash
cd deployment
python3 deployment.py \
  --imu-min-count 15 \
  --imu-max-count 60
```

9. Use explicit label lists for gesture and voice outputs:

```bash
cd deployment
python3 deployment.py \
  --gesture-labels '0,1,2,3,4,5' \
  --voice-labels '0,1,2'
```

10. Voice preprocessing with a custom sample rate, spool directory, and `ffmpeg` path:

```bash
cd deployment
python3 deployment.py \
  --voice-topic 'phone/+/viz/mic' \
  --voice-sample-rate 16000 \
  --voice-spool-dir ./spool \
  --ffmpeg-path /usr/bin/ffmpeg
```

11. Keep temporary voice files for debugging:

```bash
cd deployment
python3 deployment.py \
  --voice-topic 'phone/+/viz/mic' \
  --keep-voice-files
```

12. MQTT QoS tuning:

```bash
cd deployment
python3 deployment.py \
  --subscribe-qos 1 \
  --publish-qos 1
```

13. TLS with client certificate and insecure verification disabled:

```bash
cd deployment
python3 deployment.py \
  --cafile /home/xilinx/ca.crt \
  --certfile /home/xilinx/client.crt \
  --keyfile /home/xilinx/client.key
```

14. TLS with insecure verification enabled for test setups only:

```bash
cd deployment
python3 deployment.py \
  --cafile /home/xilinx/ca.crt \
  --tls-insecure
```

15. A fuller deployment example combining the main runtime options:

```bash
cd deployment
python3 deployment.py \
  --host 13.238.81.254 \
  --port 8883 \
  --username mqttuser \
  --password cg4002 \
  --cafile /home/xilinx/ca.crt \
  --imu-topic 'esp32/+/sensor/imu' \
  --voice-topic 'phone/+/viz/mic' \
  --action-topic 'ultra96/ai/action' \
  --pokemon-topic 'ultra96/ai/pokemon' \
  --error-topic 'ultra96/ai/error' \
  --xsa-path dual_cnn.xsa \
  --gesture-core gesture_cnn_0 \
  --voice-core voice_cnn_0 \
  --gesture-dma axi_dma_1 \
  --voice-dma axi_dma_0 \
  --timeout-s 2.0 \
  --ffmpeg-path /usr/bin/ffmpeg
```

Current default deployment settings:

- broker host: `13.238.81.254`
- broker port: `8883`
- CA file: `/home/xilinx/ca.crt`
- subscribe IMU: `esp32/+/sensor/imu`
- IMU count range: `15` to `60`, resampled to the model's fixed `60` steps
- gesture labels sent to EC2: `0`, `1`, `2`, `3`, `4`, `5`
- gesture label meanings: `0=Raise`, `1=Shake`, `2=Chop`, `3=Stir`, `4=Swing`, `5=Punch`
- voice labels sent to EC2: `0`, `1`, `2`
- voice label meanings: `0=bulbasaur`, `1=charizard`, `2=pikachu`
- publish action: `ultra96/ai/action`
- publish pokemon: `ultra96/ai/pokemon`
- publish errors: `ultra96/ai/error`

Voice is disabled by default until `phone/<player>/viz/mic` is deployed.

## Notes

- each subfolder is intended to run on its own from within that folder
- `deployment/` is self-contained and does not depend on sibling folders
- `mqtt-test/` is only for smoke testing
- `local-ai-test/` is only for offline/local evaluation
