from __future__ import annotations

"""End-to-end MQTT smoke test for the Ultra96 deployment bridge."""

import argparse
import json
import threading

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:
    raise SystemExit("The `paho-mqtt` package is required for ultra96/mqtt-test/self_test.py") from exc

from common import (
    DEFAULT_ACTION_TOPIC,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_CAFILE,
    DEFAULT_PASSWORD,
    DEFAULT_USERNAME,
    default_client_id,
    reason_code_value,
)


def _make_test_imu_window() -> list[dict[str, float]]:
    return [
        {
            "y": float(i % 6),
            "p": float((i % 5) * 0.1),
            "r": float((i % 4) * -0.1),
            "ax": 0.01 * i,
            "ay": -0.02 * i,
            "az": 9.81,
        }
        for i in range(60)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a synthetic IMU request and wait for one Ultra96 reply.")
    parser.add_argument("--host", default=DEFAULT_BROKER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--cafile", default=DEFAULT_CAFILE)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--request-topic", default="esp32/1/sensor/imu")
    parser.add_argument("--response-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--qos", type=int, default=1, choices=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reply_event = threading.Event()

    def on_connect(client, _userdata, _flags, reason_code, _properties=None) -> None:
        code = reason_code_value(reason_code)
        if code != 0:
            raise RuntimeError(f"MQTT connect failed with reason code {code}")
        client.subscribe(args.response_topic, qos=args.qos)
        payload = {"type": "imu", "data": {"samples": _make_test_imu_window(), "count": 60}}
        client.publish(args.request_topic, json.dumps(payload, separators=(",", ":"), ensure_ascii=True), qos=args.qos)
        print(f"Published test IMU request to {args.request_topic}")

    def on_message(_client, _userdata, msg) -> None:
        decoded = msg.payload.decode("utf-8", errors="replace")
        print(f"Received reply on {msg.topic}: {decoded}")
        reply_event.set()

    client = mqtt.Client(client_id=default_client_id("ultra96-selftest"))
    client.tls_set(ca_certs=args.cafile)
    client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    try:
        if not reply_event.wait(timeout=args.timeout_s):
            raise TimeoutError(f"No reply received on {args.response_topic} within {args.timeout_s:.1f}s")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
