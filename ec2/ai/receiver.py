from __future__ import annotations

"""Ultra96 -> EC2 packet receiver.

This file only accepts inference results:
- `action`
- `pokemon`
"""

import inspect
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from .messages import ClassificationData, MessageKind, Packet
from .transport import receive_json

ActionHandler = Callable[[Packet, ClassificationData], Optional[Awaitable[None]]]
PokemonHandler = Callable[[Packet, ClassificationData], Optional[Awaitable[None]]]


async def _maybe_await(result: Optional[Awaitable[None]]) -> None:
    if inspect.isawaitable(result):
        await result


class AIReceiver:
    """Dispatch validated result packets to registered callbacks."""

    def __init__(self) -> None:
        self._action_handlers: list[ActionHandler] = []
        self._pokemon_handlers: list[PokemonHandler] = []

    def on_action(self, handler: ActionHandler) -> None:
        """Register a callback for `action` packets."""

        self._action_handlers.append(handler)

    def on_pokemon(self, handler: PokemonHandler) -> None:
        """Register a callback for `pokemon` packets."""

        self._pokemon_handlers.append(handler)

    async def dispatch(self, message: Union[Mapping[str, Any], Packet]) -> None:
        """Validate one incoming packet and forward it to the correct handler list."""

        packet = message if isinstance(message, Packet) else Packet.from_dict(message)
        if packet.kind == MessageKind.ACTION:
            data = ClassificationData.from_dict(packet.data, packet.kind)
            for handler in self._action_handlers:
                await _maybe_await(handler(packet, data))
            return

        if packet.kind == MessageKind.POKEMON:
            data = ClassificationData.from_dict(packet.data, packet.kind)
            for handler in self._pokemon_handlers:
                await _maybe_await(handler(packet, data))
            return

        raise ValueError(
            f"AIReceiver only accepts '{MessageKind.ACTION.value}' or '{MessageKind.POKEMON.value}' messages"
        )

    async def listen_forever(self, websocket) -> None:
        """Continuously read JSON packets from the websocket and dispatch them."""

        while True:
            message = await receive_json(websocket)
            await self.dispatch(message)
