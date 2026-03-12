from __future__ import annotations

"""EC2 -> Ultra96 packet receiver."""

import inspect
from typing import Any, Awaitable, Callable, Mapping

try:
    from .messages import ImuData, MessageKind, Packet, VoiceMfccData
    from .transport import receive_json
except ImportError:
    from messages import ImuData, MessageKind, Packet, VoiceMfccData
    from transport import receive_json

ImuHandler = Callable[[Packet, ImuData], Awaitable[None] | None]
VoiceHandler = Callable[[Packet, VoiceMfccData], Awaitable[None] | None]


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if inspect.isawaitable(result):
        await result


class Ultra96Receiver:
    def __init__(self) -> None:
        self._imu_handlers: list[ImuHandler] = []
        self._voice_handlers: list[VoiceHandler] = []

    def on_imu(self, handler: ImuHandler) -> None:
        self._imu_handlers.append(handler)

    def on_voice_mfcc(self, handler: VoiceHandler) -> None:
        self._voice_handlers.append(handler)

    async def dispatch(self, message: Mapping[str, Any] | Packet) -> None:
        packet = message if isinstance(message, Packet) else Packet.from_dict(message)
        if packet.kind == MessageKind.IMU:
            data = ImuData.from_dict(packet.data)
            for handler in self._imu_handlers:
                await _maybe_await(handler(packet, data))
            return

        if packet.kind == MessageKind.VOICE_MFCC:
            data = VoiceMfccData.from_dict(packet.data)
            for handler in self._voice_handlers:
                await _maybe_await(handler(packet, data))
            return

        raise ValueError(
            f"Ultra96Receiver only accepts '{MessageKind.IMU.value}' or '{MessageKind.VOICE_MFCC.value}' messages"
        )

    async def listen_forever(self, websocket) -> None:
        while True:
            message = await receive_json(websocket)
            await self.dispatch(message)
