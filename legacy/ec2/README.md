# AI Comms Module

AI-side scaffold for the shared EC2 server.

Scope:
- receive `imu` messages on EC2
- preprocess voice on EC2 into MFCC
- send `imu` and `voice_mfcc` from EC2 to Ultra96
- receive `action` and `pokemon` results from Ultra96

This package is intentionally AI-only. It does not implement the central hub or the other teams' modules.

## Files

- `messages.py`: JSON envelope and payload schemas
- `receiver.py`: validates inbound `action` and `pokemon` messages from Ultra96
- `sender.py`: builds and sends outbound `imu` and `voice_mfcc` messages to Ultra96
- `service.py`: optional wrapper for sending requests and consuming results
- `audio.py`: `.m4a` decode and MFCC extraction helpers
- `transport.py`: JSON-over-WebSocket helpers
- `../server.py`: runnable EC2 WebSocket server for Ultra96 testing

## EC2 server info

Current shared EC2 server details from the comms setup:

- public IP: `13.238.81.254`
- SSH user: `ec2-user`
- access key: use the provided `.pem` file
- CA certificate: available from the shared comms folder if needed for TLS-related setup

SSH command:

```powershell
ssh -i "your\path\to\ec2-key.pem" ec2-user@13.238.81.254
```

## Protocol boundary

The shared EC2 machine also has an MQTT broker configured by the communications team, but the AI path documented in this folder does not use MQTT.

For AI specifically:
- EC2 <-> Ultra96 uses WebSocket
- EC2 -> Ultra96 sends `imu`
- EC2 -> Ultra96 sends `voice_mfcc`
- Ultra96 -> EC2 sends `action`
- Ultra96 -> EC2 sends `pokemon`

The MQTT examples from the comms team are therefore infrastructure notes for the shared server, not the packet contract for the AI-Ultra96 link.

## Connection model

The AI link uses one dedicated WebSocket server port on EC2, with a prefixed path.

- EC2 runs `server.py` and listens on one port such as `8765`
- EC2 exposes the AI bridge on a path such as `/ai/ws`
- Ultra96 runs `ultra96/ec2_bridge.py` and connects to that exact EC2 port and path
- the socket carries only AI bridge packets:
  - EC2 -> Ultra96: `imu`, `voice_mfcc`
  - Ultra96 -> EC2: `action`, `pokemon`

This is separate from the MQTT broker on port `8883`.

## Current deployment paths

For the current working setup used in team testing:

- EC2 folder: `~/ai-test`
- Ultra96 folder: `/home/xilinx/ec2-ultra96`

All commands below use those paths directly.

## Runnable EC2 server

The repo now includes a concrete EC2 entrypoint at `ec2/server.py`.

Install dependencies on EC2:

```bash
python3 -m pip install websockets numpy
```

If you want to test voice end-to-end from `.m4a`, also ensure `ffmpeg` is installed on EC2.

Start the EC2 server:

```bash
cd ~/ai-test
python3 server.py --host 0.0.0.0 --port 8765 --path /ai/ws --send-test-imu
```

Example with both IMU and voice test packets:

```bash
cd ~/ai-test
python3 server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --path /ai/ws \
  --send-test-imu \
  --m4a-path sample.m4a
```

Notes:
- `--send-test-imu` sends one built-in IMU packet right after Ultra96 connects
- `--imu-json path/to/file.json` can replace the built-in IMU packet with your own JSON array of samples
- `--m4a-path sample.m4a` makes EC2 decode audio and send one `voice_mfcc` packet after Ultra96 connects
- the current Ultra96 runtime expects exactly `60` IMU samples for a gesture inference request

## Direct connection status

The intended architecture is still:

- EC2 hosts the WebSocket server
- Ultra96 acts as the WebSocket client

However, in the current network environment, the Ultra96 could not open direct TCP
connections to the EC2 public IP. In practice:

- laptop -> EC2 worked
- Ultra96 -> EC2 timed out on both `22` and `8765`

Because of that, the currently working solution uses the laptop as an SSH relay while
the laptop is connected to the NUS SoC VPN.

## Working tunnel topology

The working path is:

```text
Ultra96 bridge client
-> ws://127.0.0.1:9000/ai/ws
-> SSH remote forward on the laptop
-> 13.238.81.254:8765
-> EC2 WebSocket server at /ai/ws
```

In other words:

1. EC2 runs the AI WebSocket server on `13.238.81.254:8765`
2. the laptop opens an SSH session to Ultra96 with a remote port forward
3. that remote port appears on Ultra96 as `127.0.0.1:9000`
4. Ultra96 connects to `127.0.0.1:9000`
5. the SSH tunnel carries that traffic through the laptop to EC2

This avoids the direct Ultra96 -> EC2 network block without changing the application
packet contract.

## Testing sequence

Use the 3-terminal setup below. This is the current working procedure.

Use this order. Do not start with Ultra96 first unless the EC2 server is already listening.

### Terminal 1: EC2

SSH into EC2 and start the AI WebSocket server:

```bash
ssh -i /path/to/ec2-key.pem ec2-user@13.238.81.254
cd ~/ai-test
python3 -m pip install websockets numpy
python3 server.py --host 0.0.0.0 --port 8765 --path /ai/ws --send-test-imu
```

Expected EC2 startup output:

```text
EC2 AI bridge listening on ws://0.0.0.0:8765/ai/ws
This port is dedicated to the Ultra96 AI WebSocket link.
```

### Terminal 2: Laptop on NUS SoC VPN

Keep the laptop connected to the NUS SoC VPN, then open the SSH tunnel to Ultra96:

```bash
ssh -N -R 9000:13.238.81.254:8765 xilinx@makerslab-fpga-39.ddns.comp.nus.edu.sg
```

Important notes:

- this command is expected to stay in the foreground
- it will appear to "hang" after login because `-N` means "tunnel only"
- do not close this terminal while testing
- this command creates `127.0.0.1:9000` on the Ultra96 side
- traffic sent by Ultra96 to `127.0.0.1:9000` is carried through the laptop to EC2 `13.238.81.254:8765`

### Terminal 3: Ultra96

On Ultra96, first optionally confirm the forwarded local port exists:

```bash
nc -vz -w 5 127.0.0.1 9000
```

Then start the Ultra96 bridge:

```bash
cd /home/xilinx/ec2-ultra96
python3 -m pip install websockets
python3 ec2_bridge.py --ws-url ws://127.0.0.1:9000/ai/ws --xsa-path dual_cnn.xsa
```

Expected Ultra96 output:

```text
Connecting to EC2 websocket: ws://127.0.0.1:9000/ai/ws
Connected. Waiting for EC2 requests.
```

### Watch the EC2 terminal

Expected behavior:
- EC2 prints that Ultra96 connected
- if `--send-test-imu` was used, EC2 sends one `imu` packet
- Ultra96 runs the gesture accelerator and replies with one `action` packet
- EC2 prints the returned action label and confidence

Example of a successful IMU test:

```text
Ultra96 connected: (...) path=/ai/ws
Sending IMU packet with 60 samples
ACTION label=3 confidence=1.0000
```

### Voice test

To test voice as well, restart the EC2 server with `--m4a-path sample.m4a`, then repeat the Ultra96 connection.

Terminal 1 on EC2:

```bash
cd ~/ai-test
python3 server.py --host 0.0.0.0 --port 8765 --path /ai/ws --send-test-imu --m4a-path sample.m4a
```

Expected behavior:
- EC2 converts `.m4a` to MFCC `[40, 50]`
- EC2 sends one `voice_mfcc` packet
- Ultra96 runs the voice accelerator and replies with one `pokemon` packet
- EC2 prints the returned label and confidence

### Stopping the setup

- EC2 server terminal: `Ctrl+C`
- laptop SSH tunnel terminal: `Ctrl+C`
- Ultra96 bridge terminal: `Ctrl+C`

### Confidence note

The current `confidence` returned by the Ultra96 bridge is a placeholder value.

Reason:

- the deployed overlay currently returns only the predicted class index
- the bridge does not yet receive logits or calibrated probabilities from the hardware path
- therefore the bridge currently sends a fixed confidence value for schema compatibility

Do not interpret `confidence=1.0000` as a real model score.

## End-to-end summary

```text
1. EC2 server listens on ws://0.0.0.0:8765/ai/ws
2. laptop opens ssh -N -R 9000:13.238.81.254:8765 to Ultra96
3. Ultra96 connects to ws://127.0.0.1:9000/ai/ws
4. the SSH tunnel relays traffic through the laptop to EC2
5. EC2 sends imu and/or voice_mfcc
6. Ultra96 runs hardware inference
7. Ultra96 sends action and/or pokemon back to EC2
8. EC2 prints the result in its terminal
```

## Shared comms note

The communication teammate provided these infrastructure notes for the shared EC2 server:

- the EC2 public IP is locked to `13.238.81.254`
- the server is intended to be left running for team testing
- SSH access uses the provided `.pem` key
- the server has an MQTT broker with TLS material under `/opt/mosquitto/certs`

Reference commands they shared:

Subscribe:

```bash
docker run --rm --init \
  --network host \
  -v /opt/mosquitto/certs:/certs \
  eclipse-mosquitto \
  mosquitto_sub -h 13.238.81.254  -p 8883 -t "system/prod/esp32/#" \
  -u mqttuser -P CS4204 \
  --cafile /certs/ca.crt
```

Publish:

```bash
docker run --rm \
  --network host \
  -v /opt/mosquitto/certs:/certs \
  eclipse-mosquitto \
  mosquitto_pub -h 13.238.81.254 -p 8883 -t "system/prod/esp32/esp32-01/cmd" \
  -m "[0.1,0.5,0.8,0.2,0.9,0.4,0.7]" \
  -u mqttuser -P CS4204 \
  --cafile /certs/ca.crt
```

These commands are kept here for context only. They are not used by the AI EC2 <-> Ultra96 code in this folder.

## EC2 <-> Ultra96 WebSocket schema

Use one JSON packet per event. Do not combine input and output into the same message.

### EC2 -> Ultra96

`imu`

```json
{
  "type": "imu",
  "data": {
    "samples": [
      {
        "y": 12.34,
        "p": 5.67,
        "r": -1.23,
        "ax": 0.01,
        "ay": -0.05,
        "az": 9.81
      }
    ],
    "count": 1
  }
}
```

`voice_mfcc`

```json
{
  "type": "voice_mfcc",
  "data": {
    "shape": [40, 50],
    "features": [
      [0.12, -0.44, 0.31, 0.08, 0.15, 0.22, 0.19, -0.01, 0.05, 0.07, 0.11, 0.03, -0.09, 0.14, 0.18, 0.21, 0.17, 0.09, 0.04, -0.02, 0.01, 0.06, 0.08, 0.12, 0.13, 0.16, 0.2, 0.24, 0.27, 0.25, 0.22, 0.18, 0.1, 0.02, -0.03, -0.07, -0.1, -0.05, 0.0, 0.04, 0.09, 0.13, 0.15, 0.18, 0.2, 0.23, 0.19, 0.11, 0.05, 0.01],
      [0.08, -0.29, 0.27, 0.04, 0.12, 0.19, 0.16, 0.0, 0.03, 0.05, 0.09, 0.02, -0.07, 0.1, 0.14, 0.18, 0.13, 0.07, 0.02, -0.01, 0.0, 0.03, 0.07, 0.1, 0.11, 0.13, 0.17, 0.21, 0.24, 0.22, 0.19, 0.15, 0.08, 0.01, -0.02, -0.05, -0.08, -0.03, 0.01, 0.03, 0.07, 0.11, 0.13, 0.16, 0.18, 0.2, 0.16, 0.09, 0.03, 0.0]
    ]
  }
}
```

Notes:
- `shape` is fixed at `[40, 50]`
- `features` must contain the full `40 x 50` MFCC matrix
- the matrix above is truncated for readability, the real packet must contain all rows
- `.m4a` should be decoded and converted to MFCC on EC2 before sending to Ultra96

### Ultra96 -> EC2

`action`

```json
{
  "type": "action",
  "data": {
    "label": "3",
    "confidence": 0.81
  }
}
```

`pokemon`

```json
{
  "type": "pokemon",
  "data": {
    "label": "0",
    "confidence": 0.88
  }
}
```

## Compact summary

```text
EC2 -> Ultra96

1. imu
{
  "type": "imu",
  "data": {
    "samples": [
      {
        "y": number,
        "p": number,
        "r": number,
        "ax": number,
        "ay": number,
        "az": number
      }
    ],
    "count": integer
  }
}

2. voice_mfcc
{
  "type": "voice_mfcc",
  "data": {
    "shape": [40, 50],
    "features": number[40][50]
  }
}

Ultra96 -> EC2

3. action
{
  "type": "action",
  "data": {
    "label": "0" | "1" | "2" | "3" | "4" | "5",
    "confidence": number
  }
}

4. pokemon
{
  "type": "pokemon",
  "data": {
    "label": "0" | "1" | "2",
    "confidence": number
  }
}
```

Current label meanings:
- Gesture/action: `0=Raise`, `1=Shake`, `2=Chop`, `3=Stir`, `4=Swing`, `5=Punch`
- Voice/pokemon: `0=bulbasaur`, `1=charizard`, `2=pikachu`

## Previous scaffold note

The Python helper files in this folder now reflect the packet contract above directly: `type` plus `data`, with separate messages for `imu`, `voice_mfcc`, `action`, and `pokemon`.

## Minimal usage

```python
import asyncio

from ec2.ai.audio import m4a_to_mfcc_matrix
from ec2.ai.receiver import AIReceiver
from ec2.ai.sender import AISender
from ec2.ai.service import AIService


async def handle_action(result):
    print("action:", result.label, result.confidence)


async def handle_pokemon(result):
    print("pokemon:", result.label, result.confidence)


async def main():
    websocket = ...  # provide your websocket client here
    receiver = AIReceiver()
    sender = AISender(websocket)
    service = AIService(sender, receiver, handle_action, handle_pokemon)

    await sender.send_imu(
        [
            {"y": 12.34, "p": 5.67, "r": -1.23, "ax": 0.01, "ay": -0.05, "az": 9.81}
        ]
    )

    mfcc = m4a_to_mfcc_matrix("sample.m4a", ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe")
    await sender.send_voice_mfcc(mfcc.tolist())

    await service.receive_forever(websocket)


asyncio.run(main())
```

## Audio helper

Use `m4a_to_mfcc_matrix()` when Viz provides `.m4a` audio:

```python
from ec2.ai.audio import m4a_to_mfcc_matrix

mfcc = m4a_to_mfcc_matrix("sample.m4a", ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe")
```

Requirements:
- `ffmpeg` must be installed on the EC2 host, or you must pass the full path to the executable
- output MFCC shape is always `[40, 50]`
