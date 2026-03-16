from __future__ import annotations

"""Tiny JSON-over-WebSocket helpers.

The package stays library-agnostic here: any websocket object with `send()` and
`recv()` methods can be used.
"""

import json
from typing import Any, Mapping, Protocol, Union


class WebSocketLike(Protocol):
    """Structural type for websocket clients used by this package."""

    async def send(self, message: str) -> Any: ...

    async def recv(self) -> Union[str, bytes]: ...


async def send_json(websocket: WebSocketLike, message: Mapping[str, Any]) -> None:
    """Serialize a Python mapping to compact JSON and send it over the websocket."""

    await websocket.send(json.dumps(dict(message), separators=(",", ":"), ensure_ascii=True))


async def receive_json(websocket: WebSocketLike) -> dict[str, Any]:
    """Receive one websocket frame and parse it as JSON."""

    raw_message = await websocket.recv()
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    return json.loads(raw_message)
