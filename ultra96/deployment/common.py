from __future__ import annotations

"""Shared helpers for the Ultra96 MQTT AI deployment and self-test scripts."""

import base64
import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
import hardware as hw

DEFAULT_XSA_PATH = (SCRIPT_DIR / "dual_cnn.xsa").resolve()
DEFAULT_SPOOL_DIR = SCRIPT_DIR / "spool"
DEFAULT_BROKER_HOST = "13.238.81.254"
DEFAULT_BROKER_PORT = 8883
DEFAULT_CAFILE = "/home/xilinx/ca.crt"
DEFAULT_USERNAME = "mqttuser"
DEFAULT_PASSWORD = "cg4002"
DEFAULT_IMU_TOPIC = "esp32/+/sensor/imu"
DEFAULT_VOICE_TOPIC = None
DEFAULT_ACTION_TOPIC = "ultra96/ai/action"
DEFAULT_POKEMON_TOPIC = "ultra96/ai/pokemon"
DEFAULT_ERROR_TOPIC = "ultra96/ai/error"


def default_client_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def parse_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required")
    return labels


def try_parse_json(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def extract_audio_bytes(data: Mapping[str, Any]) -> Optional[bytes]:
    for key in ("audio_base64", "file_base64", "payload_base64", "content_base64"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return base64.b64decode(value)
    return None


def normalize_suffix(value: str) -> str:
    suffix = value.strip() or ".m4a"
    if not suffix.startswith("."):
        suffix = "." + suffix
    return suffix


def reason_code_value(reason_code: Any) -> int:
    if hasattr(reason_code, "value"):
        return int(reason_code.value)
    return int(reason_code)


def extract_player_id(topic: str) -> Optional[str]:
    parts = [part for part in topic.split("/") if part]
    if len(parts) < 4:
        return None
    if parts[2] not in {"sensor", "viz"}:
        return None
    return parts[1]


class SuppressFileErrors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return True


class SuppressMqttErrors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return True
