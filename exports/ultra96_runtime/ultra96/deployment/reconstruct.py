from __future__ import annotations

"""Helpers for reassembling chunked voice payloads received over MQTT."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(slots=True)
class ReconstructedAudio:
    message_id: str
    received_chunks: int
    total_chunks: int
    audio_bytes: Optional[bytes] = None
    suffix: str = ".wav"

    @property
    def is_complete(self) -> bool:
        return self.audio_bytes is not None


@dataclass(slots=True)
class _PartialAudio:
    total_chunks: int
    suffix: str
    chunks: dict[int, bytes]
    updated_at: float


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


class VoiceChunkReconstructor:
    def __init__(self, default_suffix: str = ".wav", max_age_s: float = 30.0) -> None:
        self.default_suffix = self._normalize_suffix(default_suffix)
        self.max_age_s = float(max_age_s)
        self._messages: dict[str, _PartialAudio] = {}

    @staticmethod
    def is_chunk_payload(value: Mapping[str, Any]) -> bool:
        return all(key in value for key in ("id", "index", "total", "data"))

    def add_chunk(self, payload: Mapping[str, Any]) -> ReconstructedAudio:
        if not self.is_chunk_payload(payload):
            raise ValueError("Chunk payload must include id, index, total, and data fields")

        self._evict_expired()

        message_id = _require_non_empty_string(payload.get("id"), "voice_chunk.id")
        index = _require_int(payload.get("index"), "voice_chunk.index")
        total_chunks = _require_int(payload.get("total"), "voice_chunk.total")
        data_hex = _require_non_empty_string(payload.get("data"), "voice_chunk.data")

        if total_chunks <= 0:
            raise ValueError("voice_chunk.total must be greater than zero")
        if index < 0 or index >= total_chunks:
            raise ValueError(
                f"voice_chunk.index must be between 0 and total-1, got index={index}, total={total_chunks}"
            )

        try:
            chunk_bytes = bytes.fromhex(data_hex)
        except ValueError as exc:
            raise ValueError("voice_chunk.data must be a valid hex string") from exc

        suffix = self._resolve_suffix(payload)
        now = time.monotonic()

        message = self._messages.get(message_id)
        if message is None:
            message = _PartialAudio(
                total_chunks=total_chunks,
                suffix=suffix,
                chunks={},
                updated_at=now,
            )
            self._messages[message_id] = message
        else:
            if message.total_chunks != total_chunks:
                raise ValueError(
                    f"voice_chunk.total changed for id={message_id}: {message.total_chunks} -> {total_chunks}"
                )
            if message.suffix != suffix:
                raise ValueError(f"voice chunk suffix changed for id={message_id}: {message.suffix} -> {suffix}")

        message.chunks[index] = chunk_bytes
        message.updated_at = now

        received_chunks = len(message.chunks)
        if received_chunks < total_chunks:
            return ReconstructedAudio(
                message_id=message_id,
                received_chunks=received_chunks,
                total_chunks=total_chunks,
                suffix=message.suffix,
            )

        audio_bytes = b"".join(message.chunks[idx] for idx in range(total_chunks))
        del self._messages[message_id]
        return ReconstructedAudio(
            message_id=message_id,
            received_chunks=total_chunks,
            total_chunks=total_chunks,
            audio_bytes=audio_bytes,
            suffix=message.suffix,
        )

    def _evict_expired(self) -> None:
        if self.max_age_s <= 0:
            return
        now = time.monotonic()
        expired_ids = [
            message_id
            for message_id, message in self._messages.items()
            if now - message.updated_at > self.max_age_s
        ]
        for message_id in expired_ids:
            del self._messages[message_id]

    def _resolve_suffix(self, payload: Mapping[str, Any]) -> str:
        explicit_suffix = payload.get("suffix") or payload.get("extension")
        if isinstance(explicit_suffix, str) and explicit_suffix.strip():
            return self._normalize_suffix(explicit_suffix)

        filename = payload.get("filename") or payload.get("name")
        if isinstance(filename, str) and filename.strip():
            return self._normalize_suffix(Path(filename).suffix or self.default_suffix)

        return self.default_suffix

    @staticmethod
    def _normalize_suffix(value: str) -> str:
        suffix = value.strip() or ".wav"
        if not suffix.startswith("."):
            suffix = "." + suffix
        return suffix
