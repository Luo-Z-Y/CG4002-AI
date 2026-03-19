from __future__ import annotations

"""Always-on MQTT deployment bridge for Ultra96 inference."""

import argparse
import json
import queue
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:
    raise SystemExit("The `paho-mqtt` package is required for ultra96/deployment/deployment.py") from exc

from common import (
    DEFAULT_ACTION_TOPIC,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_CAFILE,
    DEFAULT_ERROR_TOPIC,
    DEFAULT_IMU_TOPIC,
    DEFAULT_PASSWORD,
    DEFAULT_POKEMON_TOPIC,
    DEFAULT_SPOOL_DIR,
    DEFAULT_USERNAME,
    DEFAULT_VOICE_TOPIC,
    DEFAULT_XSA_PATH,
    SuppressFileErrors,
    SuppressMqttErrors,
    default_client_id,
    extract_audio_bytes,
    extract_player_id,
    normalize_suffix,
    parse_labels,
    reason_code_value,
    try_parse_json,
    hw,
)
from audio import VoicePreprocessor, m4a_to_mfcc_matrix
from imu import ImuPreprocessor
from messages import ClassificationData, ImuData, MessageKind, Packet, VoiceMfccData, build_imu_data
from runtime import Ultra96Runtime


@dataclass(slots=True)
class InboundMessage:
    topic: str
    payload: bytes


class MqttAiBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = Ultra96Runtime(
            xsa_path=args.xsa_path,
            gesture_core_name=args.gesture_core,
            voice_core_name=args.voice_core,
            gesture_dma_name=args.gesture_dma,
            voice_dma_name=args.voice_dma,
            timeout_s=args.timeout_s,
            default_confidence=args.default_confidence,
            gesture_labels=parse_labels(args.gesture_labels),
            voice_labels=parse_labels(args.voice_labels),
        )
        self.voice_preprocessor = VoicePreprocessor(sample_rate=args.voice_sample_rate)
        self.imu_preprocessor = ImuPreprocessor(
            min_count=args.imu_min_count,
            max_count=args.imu_max_count,
        )
        self.queue: "queue.Queue[InboundMessage]" = queue.Queue()
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.connect_error: Optional[str] = None
        self.spool_dir = Path(args.voice_spool_dir).resolve()
        self.spool_dir.mkdir(parents=True, exist_ok=True)

        self.client = mqtt.Client(client_id=args.client_id)
        if args.username:
            self.client.username_pw_set(args.username, args.password)
        if args.cafile or args.certfile or args.keyfile:
            self.client.tls_set(ca_certs=args.cafile, certfile=args.certfile, keyfile=args.keyfile)
            if args.tls_insecure:
                self.client.tls_insecure_set(True)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def run(self) -> None:
        self.client.connect(self.args.host, self.args.port, keepalive=self.args.keepalive)
        self.client.loop_start()
        try:
            if not self.connected_event.wait(timeout=self.args.connect_timeout_s):
                raise TimeoutError(
                    f"Timed out after {self.args.connect_timeout_s:.1f}s waiting for MQTT connect to "
                    f"{self.args.host}:{self.args.port}"
                )
            if self.connect_error is not None:
                raise RuntimeError(self.connect_error)
            while not self.stop_event.is_set():
                try:
                    message = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._process_message(message)
        finally:
            self.stop_event.set()
            self.client.loop_stop()
            with SuppressMqttErrors():
                self.client.disconnect()
            self.runtime.close()

    def stop(self) -> None:
        self.stop_event.set()

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties=None) -> None:
        code = reason_code_value(reason_code)
        if code != 0:
            self.connect_error = f"MQTT connect failed with reason code {code}"
            self.stop_event.set()
            self.connected_event.set()
            return
        print(f"Connected to MQTT broker at {self.args.host}:{self.args.port}")
        client.subscribe(self.args.imu_topic, qos=self.args.subscribe_qos)
        print(f"Subscribed to IMU topic: {self.args.imu_topic}")
        if self.args.voice_topic:
            client.subscribe(self.args.voice_topic, qos=self.args.subscribe_qos)
            print(f"Subscribed to voice topic: {self.args.voice_topic}")
        else:
            print("Voice topic disabled")
        self.connected_event.set()

    def _on_disconnect(self, _client, _userdata, reason_code, _properties=None) -> None:
        code = reason_code_value(reason_code)
        if self.stop_event.is_set():
            print("MQTT bridge stopped")
        else:
            print(f"[WARN] MQTT disconnected with reason code {code}")

    def _on_message(self, _client, _userdata, message) -> None:
        self.queue.put(InboundMessage(topic=message.topic, payload=bytes(message.payload)))

    def _process_message(self, message: InboundMessage) -> None:
        player_id = extract_player_id(message.topic)
        try:
            if mqtt.topic_matches_sub(self.args.imu_topic, message.topic):
                data = self._parse_imu_payload(message.payload)
                data = self.imu_preprocessor.preprocess(data)
                result = self.runtime.classify_imu(data)
                self._publish_result(self.args.action_topic, MessageKind.ACTION, result, player_id, message.topic)
                print(
                    f"IMU player={player_id or 'unknown'} -> action={result.label} "
                    f"confidence={result.confidence:.4f}"
                )
                return

            if self.args.voice_topic and mqtt.topic_matches_sub(self.args.voice_topic, message.topic):
                data = self._parse_voice_payload(message.payload)
                result = self.runtime.classify_voice_mfcc(data)
                self._publish_result(self.args.pokemon_topic, MessageKind.POKEMON, result, player_id, message.topic)
                print(
                    f"VOICE player={player_id or 'unknown'} -> pokemon={result.label} "
                    f"confidence={result.confidence:.4f}"
                )
                return

            raise ValueError(f"Received MQTT message on unexpected topic: {message.topic}")
        except Exception as exc:
            self._publish_error(message.topic, str(exc), player_id)
            print(f"[ERROR] topic={message.topic} error={exc}")

    @staticmethod
    def _parse_imu_payload(payload: bytes):
        parsed = try_parse_json(payload)
        if isinstance(parsed, list):
            return build_imu_data(parsed)
        if not isinstance(parsed, dict):
            raise TypeError("IMU payload must be JSON")

        packet_type = parsed.get("type")
        packet_data = parsed.get("data")
        if packet_type == MessageKind.IMU.value and isinstance(packet_data, dict):
            return ImuData.from_dict(packet_data)

        candidate = packet_data if isinstance(packet_data, dict) else parsed
        if "samples" in candidate:
            return ImuData.from_dict(candidate)

        raise ValueError("Unsupported IMU payload format")

    def _parse_voice_payload(self, payload: bytes) -> VoiceMfccData:
        parsed = try_parse_json(payload)
        if isinstance(parsed, dict):
            packet_type = parsed.get("type")
            packet_data = parsed.get("data")
            if packet_type == MessageKind.VOICE_MFCC.value and isinstance(packet_data, dict):
                return VoiceMfccData.from_dict(packet_data)

            candidate = packet_data if isinstance(packet_data, dict) else parsed
            if "features" in candidate:
                return VoiceMfccData.from_dict(candidate)

            audio_bytes = extract_audio_bytes(candidate)
            if audio_bytes is not None:
                filename = str(candidate.get("filename") or candidate.get("name") or "")
                suffix = Path(filename).suffix or self.args.voice_file_extension
                return self._mfcc_from_file_bytes(audio_bytes, suffix)

        return self._mfcc_from_file_bytes(payload, self.args.voice_file_extension)

    def _mfcc_from_file_bytes(self, payload: bytes, suffix: str) -> VoiceMfccData:
        temp_path = self.spool_dir / f"voice-{uuid.uuid4().hex}{normalize_suffix(suffix)}"
        temp_path.write_bytes(payload)
        try:
            mfcc = m4a_to_mfcc_matrix(
                temp_path,
                sample_rate=self.args.voice_sample_rate,
                ffmpeg_path=self.args.ffmpeg_path,
                preprocessor=self.voice_preprocessor,
            )
            return VoiceMfccData.from_dict(
                {"shape": [int(mfcc.shape[0]), int(mfcc.shape[1])], "features": mfcc.tolist()}
            )
        finally:
            if not self.args.keep_voice_files:
                with SuppressFileErrors():
                    temp_path.unlink()

    def _publish_result(
        self,
        topic: str,
        kind: MessageKind,
        result: ClassificationData,
        player_id: Optional[str],
        source_topic: str,
    ) -> None:
        packet = Packet(kind=kind, data=result.to_dict()).to_dict()
        if player_id is not None:
            packet["player"] = player_id
        packet["source_topic"] = source_topic
        self.client.publish(topic, json.dumps(packet, separators=(",", ":"), ensure_ascii=True), qos=self.args.publish_qos)

    def _publish_error(self, source_topic: str, message: str, player_id: Optional[str]) -> None:
        if not self.args.error_topic:
            return
        payload = {"type": "error", "data": {"source_topic": source_topic, "message": message}}
        if player_id is not None:
            payload["player"] = player_id
        self.client.publish(
            self.args.error_topic,
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            qos=self.args.publish_qos,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to MQTT IMU/voice topics and run Ultra96 inference.")
    parser.add_argument("--host", default=DEFAULT_BROKER_HOST, help="MQTT broker host.")
    parser.add_argument("--port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--client-id", default=default_client_id("ultra96-ai"))
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--keepalive", type=int, default=60)
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--subscribe-qos", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--publish-qos", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--imu-topic", default=DEFAULT_IMU_TOPIC)
    parser.add_argument("--imu-min-count", type=int, default=15)
    parser.add_argument("--imu-max-count", type=int, default=60)
    parser.add_argument("--voice-topic", default=DEFAULT_VOICE_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--pokemon-topic", default=DEFAULT_POKEMON_TOPIC)
    parser.add_argument("--error-topic", default=DEFAULT_ERROR_TOPIC)
    parser.add_argument("--cafile", default=DEFAULT_CAFILE)
    parser.add_argument("--certfile", default=None)
    parser.add_argument("--keyfile", default=None)
    parser.add_argument("--tls-insecure", action="store_true")
    parser.add_argument("--voice-file-extension", default=".m4a")
    parser.add_argument("--voice-sample-rate", type=int, default=16000)
    parser.add_argument("--voice-spool-dir", default=str(DEFAULT_SPOOL_DIR))
    parser.add_argument("--keep-voice-files", action="store_true")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--xsa-path", default=str(DEFAULT_XSA_PATH))
    parser.add_argument("--gesture-core", default=hw.GESTURE_CORE_NAME)
    parser.add_argument("--voice-core", default=hw.VOICE_CORE_NAME)
    parser.add_argument("--gesture-dma", default=hw.GESTURE_DMA_NAME)
    parser.add_argument("--voice-dma", default=hw.VOICE_DMA_NAME)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--default-confidence", type=float, default=1.0)
    parser.add_argument("--voice-labels", default=",".join(hw.VOICE_LABELS))
    parser.add_argument("--gesture-labels", default=",".join(hw.GESTURE_LABELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge = MqttAiBridge(args)

    def _handle_signal(_signum, _frame) -> None:
        bridge.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    bridge.run()


if __name__ == "__main__":
    main()
