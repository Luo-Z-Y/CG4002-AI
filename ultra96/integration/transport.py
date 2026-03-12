from __future__ import annotations

"""Tiny JSON-over-WebSocket helpers for the Ultra96 bridge."""

import json
from typing import Any, Mapping, Protocol


class WebSocketLike(Protocol):
    async def send(self, message: str) -> Any: ...

    async def recv(self) -> str | bytes: ...


async def send_json(websocket: WebSocketLike, message: Mapping[str, Any]) -> None:
    await websocket.send(json.dumps(dict(message), separators=(",", ":"), ensure_ascii=True))


async def receive_json(websocket: WebSocketLike) -> dict[str, Any]:
    raw_message = await websocket.recv()
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    return json.loads(raw_message)
