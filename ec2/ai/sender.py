from __future__ import annotations

"""EC2 -> Ultra96 packet sender.

This file only handles outbound requests:
- `imu`
- `voice_mfcc`
"""

from pathlib import Path
from typing import Mapping, Sequence

from .audio import m4a_to_mfcc_matrix
from .messages import ImuData, ImuSample, MessageKind, Packet, VoiceMfccData, build_imu_data, build_voice_mfcc_data
from .transport import send_json


class AISender:
    """Small helper around a websocket client for sending validated packets."""

    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def send_imu(self, samples: Sequence[ImuSample | Mapping[str, float]]) -> None:
        """Send one `imu` request packet to Ultra96."""

        data = build_imu_data(samples)
        await self._send(MessageKind.IMU, data)

    async def send_voice_mfcc(self, features: Sequence[Sequence[float]] | VoiceMfccData) -> None:
        """Send one `voice_mfcc` request packet to Ultra96."""

        data = features if isinstance(features, VoiceMfccData) else build_voice_mfcc_data(features)
        await self._send(MessageKind.VOICE_MFCC, data)

    async def send_m4a_as_mfcc(self, m4a_path: str | Path, ffmpeg_path: str = "ffmpeg") -> VoiceMfccData:
        """Decode `.m4a`, convert it to MFCC, then send it as `voice_mfcc`.

        This keeps the audio decoding and MFCC work on EC2, which is the chosen split.
        """

        mfcc = m4a_to_mfcc_matrix(m4a_path, ffmpeg_path=ffmpeg_path)
        data = VoiceMfccData(features=mfcc.tolist())
        await self._send(MessageKind.VOICE_MFCC, data)
        return data

    async def _send(self, kind: MessageKind, data: ImuData | VoiceMfccData) -> None:
        # Always validate/normalize through `Packet` before writing to the websocket.
        packet = Packet(kind=kind, data=data.to_dict())
        await send_json(self.websocket, packet.to_dict())
