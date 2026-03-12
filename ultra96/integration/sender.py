from __future__ import annotations

"""Ultra96 -> EC2 packet sender."""

from typing import Dict, Union

try:
    from .messages import ClassificationData, MessageKind, Packet
    from .transport import send_json
except ImportError:
    from messages import ClassificationData, MessageKind, Packet
    from transport import send_json


class Ultra96Sender:
    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def send_action(self, result: Union[ClassificationData, Dict[str, object]]) -> None:
        data = result if isinstance(result, ClassificationData) else ClassificationData.from_dict(result, MessageKind.ACTION)
        await self._send(MessageKind.ACTION, data)

    async def send_pokemon(self, result: Union[ClassificationData, Dict[str, object]]) -> None:
        data = result if isinstance(result, ClassificationData) else ClassificationData.from_dict(result, MessageKind.POKEMON)
        await self._send(MessageKind.POKEMON, data)

    async def _send(self, kind: MessageKind, data: ClassificationData) -> None:
        packet = Packet(kind=kind, data=data.to_dict())
        await send_json(self.websocket, packet.to_dict())
