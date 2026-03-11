# AI Comms Module

AI-side scaffold for the shared game server.

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

## Shared comms note

The communication team provided these infrastructure notes for the shared EC2 server:

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
- the matrix above is truncated for readability; the real packet must contain all rows
- `.m4a` should be decoded and converted to MFCC on EC2 before sending to Ultra96

### Ultra96 -> EC2

`action`

```json
{
  "type": "action",
  "data": {
    "label": "stir",
    "confidence": 0.81
  }
}
```

`pokemon`

```json
{
  "type": "pokemon",
  "data": {
    "label": "pikachu",
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
    "label": string,
    "confidence": number
  }
}

4. pokemon
{
  "type": "pokemon",
  "data": {
    "label": string,
    "confidence": number
  }
}
```

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
