from __future__ import annotations

"""Convenience wrapper for the common EC2 runtime flow.

Typical usage:
1. send IMU or voice MFCC requests to Ultra96
2. keep listening for `action` / `pokemon` responses
3. forward those results to higher-level EC2 logic
"""

import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from .audio import m4a_to_mfcc_matrix
from .messages import ClassificationData, ImuSample
from .receiver import AIReceiver
from .sender import AISender

ActionResultHandler = Callable[[ClassificationData], Awaitable[None] | None]
PokemonResultHandler = Callable[[ClassificationData], Awaitable[None] | None]


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


class AIService:
    """Glue layer around `AISender` and `AIReceiver`.

    This file does not contain model logic. It only organizes request sending and
    result handling so the rest of the EC2 app can stay simple.
    """

    def __init__(
        self,
        sender: AISender,
        receiver: AIReceiver,
        action_handler: ActionResultHandler | None = None,
        pokemon_handler: PokemonResultHandler | None = None,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.action_handler = action_handler
        self.pokemon_handler = pokemon_handler

        if self.action_handler is not None:
            self.receiver.on_action(self._handle_action)
        if self.pokemon_handler is not None:
            self.receiver.on_pokemon(self._handle_pokemon)

    async def send_imu_samples(self, samples: list[ImuSample | dict[str, float]]) -> None:
        """Forward IMU samples to Ultra96."""

        await self.sender.send_imu(samples)

    async def send_voice_mfcc(self, features: list[list[float]]) -> None:
        """Forward an already-computed MFCC matrix to Ultra96."""

        await self.sender.send_voice_mfcc(features)

    async def send_m4a_file(self, m4a_path: str | Path, ffmpeg_path: str = "ffmpeg") -> list[list[float]]:
        """Decode `.m4a`, compute MFCC, send to Ultra96, and return the matrix used."""

        mfcc = m4a_to_mfcc_matrix(m4a_path, ffmpeg_path=ffmpeg_path)
        await self.sender.send_voice_mfcc(mfcc.tolist())
        return mfcc.tolist()

    async def _handle_action(self, _, data: ClassificationData) -> None:
        """Internal bridge from receiver callback to the external action handler."""

        await _maybe_await(self.action_handler(data))

    async def _handle_pokemon(self, _, data: ClassificationData) -> None:
        """Internal bridge from receiver callback to the external pokemon handler."""

        await _maybe_await(self.pokemon_handler(data))

    async def receive_forever(self, websocket) -> None:
        """Start the blocking receive loop for Ultra96 result packets."""

        await self.receiver.listen_forever(websocket)
