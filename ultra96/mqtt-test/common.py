from __future__ import annotations

"""Shared helpers for the Ultra96 MQTT AI deployment and self-test scripts."""

import uuid
from typing import Any

DEFAULT_BROKER_HOST = "13.238.81.254"
DEFAULT_BROKER_PORT = 8883
DEFAULT_CAFILE = "/home/xilinx/ca.crt"
DEFAULT_USERNAME = "mqttuser"
DEFAULT_PASSWORD = "CS4204"
DEFAULT_ACTION_TOPIC = "ultra96/ai/action"


def default_client_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def reason_code_value(reason_code: Any) -> int:
    if hasattr(reason_code, "value"):
        return int(reason_code.value)
    return int(reason_code)
