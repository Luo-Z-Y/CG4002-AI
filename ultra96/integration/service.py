from __future__ import annotations

"""Glue between websocket packets and Ultra96 hardware inference."""

try:
    from .messages import ImuData, VoiceMfccData
    from .receiver import Ultra96Receiver
    from .runtime import Ultra96Runtime
    from .sender import Ultra96Sender
except ImportError:
    from messages import ImuData, VoiceMfccData
    from receiver import Ultra96Receiver
    from runtime import Ultra96Runtime
    from sender import Ultra96Sender


class Ultra96Service:
    def __init__(self, sender: Ultra96Sender, receiver: Ultra96Receiver, runtime: Ultra96Runtime) -> None:
        self.sender = sender
        self.receiver = receiver
        self.runtime = runtime
        self.receiver.on_imu(self._handle_imu)
        self.receiver.on_voice_mfcc(self._handle_voice_mfcc)

    async def _handle_imu(self, _, data: ImuData) -> None:
        result = self.runtime.classify_imu(data)
        await self.sender.send_action(result)

    async def _handle_voice_mfcc(self, _, data: VoiceMfccData) -> None:
        result = self.runtime.classify_voice_mfcc(data)
        await self.sender.send_pokemon(result)

    async def serve_forever(self, websocket) -> None:
        await self.receiver.listen_forever(websocket)
