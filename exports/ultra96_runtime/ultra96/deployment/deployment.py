from __future__ import annotations

"""Always-on MQTT deployment bridge for Ultra96 inference."""

import argparse
import json
import queue
import signal
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import wave

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:
    raise SystemExit("The `paho-mqtt` package is required for ultra96/deployment/deployment.py") from exc

from common import (
    DEFAULT_ACTION_TOPIC,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_CAFILE,
    DEFAULT_CAPTURE_DIR,
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
    extract_device_id,
    normalize_suffix,
    parse_labels,
    reason_code_value,
    try_parse_json,
    hw,
)
from audio import VoicePreprocessor, load_feature_norm_stats, m4a_to_mfcc_matrix, normalize_feature_matrix
from imu import ImuPreprocessor, load_feature_norm_stats, normalize_window
from messages import ClassificationData, ImuData, MessageKind, Packet, VoiceMfccData, build_imu_data
from reconstruct import VoiceChunkReconstructor
from runtime import Ultra96Runtime

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VOICE_MEAN_PATH = SCRIPT_DIR / "voice_mean.npy"
DEFAULT_VOICE_STD_PATH = SCRIPT_DIR / "voice_std.npy"
DEFAULT_GESTURE_MEAN_PATH = SCRIPT_DIR / "gesture_mean.npy"
DEFAULT_GESTURE_STD_PATH = SCRIPT_DIR / "gesture_std.npy"


@dataclass(slots=True)
class InboundMessage:
    topic: str
    payload: bytes


class MqttAiBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session_name = self._resolve_session_name(args.session_name)
        self.runtime = Ultra96Runtime(
            xsa_path=args.xsa_path,
            gesture_core_name=args.gesture_core,
            voice_core_name=args.voice_core,
            gesture_dma_name=args.gesture_dma,
            voice_dma_name=args.voice_dma,
            timeout_s=args.timeout_s,
            gesture_mean=args.gesture_mean,
            gesture_std=args.gesture_std,
            default_confidence=args.default_confidence,
            gesture_labels=parse_labels(args.gesture_labels),
            voice_labels=parse_labels(args.voice_labels),
        )
        self.gesture_mean_path = Path(args.gesture_mean).resolve()
        self.gesture_std_path = Path(args.gesture_std).resolve()
        self.gesture_mean, self.gesture_std = load_feature_norm_stats(self.gesture_mean_path, self.gesture_std_path)
        self.voice_preprocessor = VoicePreprocessor(sample_rate=args.voice_sample_rate)
        self.voice_mean_path = Path(args.voice_mean).resolve()
        self.voice_std_path = Path(args.voice_std).resolve()
        self.voice_mean, self.voice_std = load_feature_norm_stats(self.voice_mean_path, self.voice_std_path)
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
        self.capture_dir = Path(args.capture_dir).resolve()
        self.capture_session_dir = self.capture_dir / self.session_name
        self.capture_raw_imu_dir = self.capture_session_dir / "raw_imu"
        self.capture_resampled_imu_dir = self.capture_session_dir / "resampled_imu"
        self.capture_raw_audio_dir = self.capture_session_dir / "raw_audio"
        self.capture_augmented_audio_dir = self.capture_session_dir / "augmented_audio"
        self.capture_mfcc_dir = self.capture_session_dir / "mfcc"
        self.capture_session_dir.mkdir(parents=True, exist_ok=True)
        self.capture_raw_imu_dir.mkdir(parents=True, exist_ok=True)
        self.capture_resampled_imu_dir.mkdir(parents=True, exist_ok=True)
        self.capture_raw_audio_dir.mkdir(parents=True, exist_ok=True)
        self.capture_augmented_audio_dir.mkdir(parents=True, exist_ok=True)
        self.capture_mfcc_dir.mkdir(parents=True, exist_ok=True)
        self.voice_reconstructor = VoiceChunkReconstructor(
            default_suffix=args.voice_chunk_extension,
            max_age_s=args.voice_chunk_timeout_s,
        )
        self._last_voice_debug_meta: Optional[str] = None
        self._last_voice_feature_stats: Optional[dict[str, object]] = None

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

    @classmethod
    def _resolve_session_name(cls, raw_value: Optional[str]) -> str:
        if raw_value and raw_value.strip():
            return cls._sanitize_component(raw_value)
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

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
        print(f"Capture session: {self.capture_session_dir}")
        print(f"Gesture normalization stats: mean={self.gesture_mean_path} std={self.gesture_std_path}")
        print(f"Voice normalization stats: mean={self.voice_mean_path} std={self.voice_std_path}")
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
        device_id = extract_device_id(message.topic)
        try:
            if mqtt.topic_matches_sub(self.args.imu_topic, message.topic):
                if self.args.debug:
                    print(self._describe_payload("IMU", message.topic, message.payload))
                raw_data = self._parse_imu_payload(message.payload)
                raw_count = raw_data.count if raw_data.count is not None else len(raw_data.samples)
                data = self.imu_preprocessor.preprocess(raw_data)
                self._capture_imu_sample(message.topic, device_id, message.payload, raw_data, data)
                if self.args.debug:
                    print(self._describe_imu(message.topic, raw_count, data))
                result = self.runtime.classify_imu(data)
                self._publish_result(self.args.action_topic, MessageKind.ACTION, result, device_id, message.topic)
                print(
                    f"IMU device={device_id or 'unknown'} -> action={result.label} "
                    f"confidence={result.confidence:.4f}"
                )
                return

            if self.args.voice_topic and mqtt.topic_matches_sub(self.args.voice_topic, message.topic):
                if self.args.debug:
                    print(self._describe_payload("VOICE", message.topic, message.payload))
                data = self._parse_voice_payload(message.topic, device_id, message.payload)
                if data is None:
                    return
                if self.args.debug:
                    print(self._describe_voice(message.topic, data))
                result = self.runtime.classify_voice_mfcc(data)
                self._publish_result(self.args.pokemon_topic, MessageKind.POKEMON, result, device_id, message.topic)
                print(
                    f"VOICE device={device_id or 'unknown'} -> pokemon={result.label} "
                    f"confidence={result.confidence:.4f}"
                )
                return

            raise ValueError(f"Received MQTT message on unexpected topic: {message.topic}")
        except Exception as exc:
            self._publish_error(message.topic, str(exc), device_id)
            print(f"[ERROR] topic={message.topic} error={exc}")
            if self.args.debug:
                preview = message.payload[:160].decode("utf-8", errors="replace")
                print(f"[DEBUG][ERROR] topic={message.topic} payload_preview={preview!r}")

    @staticmethod
    def _format_stats(array: np.ndarray) -> str:
        return (
            f"shape={tuple(array.shape)} min={float(array.min()):.4f} max={float(array.max()):.4f} "
            f"mean={float(array.mean()):.4f} std={float(array.std()):.4f}"
        )

    @staticmethod
    def _format_debug_stats(stats: dict[str, object]) -> str:
        return (
            f"shape={stats['shape']} min={stats['min']:.4f} max={stats['max']:.4f} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f}"
        )

    @staticmethod
    def _truncate_text(text: str, limit: int = 320) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _summarize_for_debug(self, value: Any, *, depth: int = 0) -> Any:
        if depth >= 2:
            if isinstance(value, (dict, list)):
                return f"<{type(value).__name__}>"
            return value

        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if isinstance(item, str) and len(item) > 96:
                    out[key] = f"<str len={len(item)}>"
                else:
                    out[key] = self._summarize_for_debug(item, depth=depth + 1)
            return out

        if isinstance(value, list):
            if not value:
                return []
            if len(value) <= 3:
                return [self._summarize_for_debug(item, depth=depth + 1) for item in value]
            return {
                "_list_len": len(value),
                "first": self._summarize_for_debug(value[0], depth=depth + 1),
                "last": self._summarize_for_debug(value[-1], depth=depth + 1),
            }

        return value

    def _describe_payload(self, kind: str, topic: str, payload: bytes) -> str:
        parsed = try_parse_json(payload)
        if parsed is not None:
            compact = json.dumps(self._summarize_for_debug(parsed), separators=(",", ":"), ensure_ascii=True)
            return f"[DEBUG][{kind}][PAYLOAD] topic={topic} payload={self._truncate_text(compact)}"

        preview = payload[:96].decode("utf-8", errors="replace")
        return (
            f"[DEBUG][{kind}][PAYLOAD] topic={topic} bytes={len(payload)} "
            f"preview={self._truncate_text(preview)!r}"
        )

    @staticmethod
    def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        pcm16 = np.clip(samples, -1.0, 1.0)
        pcm16 = np.round(pcm16 * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

    @staticmethod
    def _sanitize_component(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
        cleaned = cleaned.strip("._")
        return cleaned or "unknown"

    def _capture_stem(self, topic: str, device_id: Optional[str]) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_device = self._sanitize_component(device_id or "unknown")
        safe_topic = self._sanitize_component(topic.replace("/", "_"))
        return f"{stamp}_device-{safe_device}_{safe_topic}"

    def _capture_imu_sample(self, topic: str, device_id: Optional[str], payload: bytes, raw_data: ImuData, data: ImuData) -> None:
        stem = self._capture_stem(topic, device_id)
        parsed = try_parse_json(payload)
        raw_record = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "device_id": device_id,
            "player": device_id,
            "raw_count": raw_data.count,
            "raw_payload": parsed if parsed is not None else payload.decode("utf-8", errors="replace"),
            "raw_data": raw_data.to_dict(),
        }
        resampled_record = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "device_id": device_id,
            "player": device_id,
            "raw_count": raw_data.count,
            "processed_count": data.count,
            "processed_data": data.to_dict(),
            "preprocess": self.imu_preprocessor.last_debug or {},
        }
        raw_json_path = self.capture_raw_imu_dir / f"{stem}.json"
        raw_npy_path = self.capture_raw_imu_dir / f"{stem}.npy"
        resampled_json_path = self.capture_resampled_imu_dir / f"{stem}.json"
        resampled_npy_path = self.capture_resampled_imu_dir / f"{stem}.npy"
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=True), encoding="utf-8")
        resampled_json_path.write_text(json.dumps(resampled_record, indent=2, ensure_ascii=True), encoding="utf-8")
        capture = self.imu_preprocessor.last_capture or {}
        raw_window = capture.get("raw_window")
        resampled_window = capture.get("resampled_window")
        if isinstance(raw_window, np.ndarray):
            np.save(raw_npy_path, raw_window.astype(np.float32))
        if isinstance(resampled_window, np.ndarray):
            np.save(resampled_npy_path, resampled_window.astype(np.float32))

    def _capture_voice_audio(
        self,
        topic: str,
        device_id: Optional[str],
        payload: bytes,
        suffix: str,
        mfcc: np.ndarray,
        raw_mfcc: Optional[np.ndarray] = None,
    ) -> None:
        stem = self._capture_stem(topic, device_id)
        normalized_suffix = normalize_suffix(suffix)
        raw_audio_path = self.capture_raw_audio_dir / f"{stem}{normalized_suffix}"
        raw_meta_path = self.capture_raw_audio_dir / f"{stem}.json"
        augmented_audio_path = self.capture_augmented_audio_dir / f"{stem}.wav"
        augmented_meta_path = self.capture_augmented_audio_dir / f"{stem}.json"
        mfcc_path = self.capture_mfcc_dir / f"{stem}.npy"
        mfcc_meta_path = self.capture_mfcc_dir / f"{stem}.json"

        raw_audio_path.write_bytes(payload)

        debug = self.voice_preprocessor.last_debug or {}
        capture = self.voice_preprocessor.last_capture or {}
        raw_metadata = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "device_id": device_id,
            "player": device_id,
            "suffix": normalized_suffix,
            "bytes": len(payload),
            "audio_file": raw_audio_path.name,
        }
        raw_meta_path.write_text(json.dumps(raw_metadata, indent=2, ensure_ascii=True), encoding="utf-8")

        augmented_waveform = capture.get("emphasized_waveform")
        if isinstance(augmented_waveform, np.ndarray):
            self._write_wav(augmented_audio_path, augmented_waveform, self.args.voice_sample_rate)

        augmented_metadata = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "device_id": device_id,
            "player": device_id,
            "audio_file": augmented_audio_path.name,
            "sample_rate": self.args.voice_sample_rate,
            "preprocess": debug,
        }
        augmented_meta_path.write_text(json.dumps(augmented_metadata, indent=2, ensure_ascii=True), encoding="utf-8")

        np.save(mfcc_path, mfcc.astype(np.float32))
        mfcc_metadata = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "device_id": device_id,
            "player": device_id,
            "mfcc_file": mfcc_path.name,
            "mfcc_shape": [int(mfcc.shape[0]), int(mfcc.shape[1])],
            "preprocess": debug,
            "normalization": {
                "mean_file": self.voice_mean_path.name,
                "std_file": self.voice_std_path.name,
                "mean_path": str(self.voice_mean_path),
                "std_path": str(self.voice_std_path),
            },
        }
        if isinstance(raw_mfcc, np.ndarray):
            mfcc_metadata["raw_mfcc_stats"] = {
                "shape": [int(raw_mfcc.shape[0]), int(raw_mfcc.shape[1])],
                "min": float(raw_mfcc.min()),
                "max": float(raw_mfcc.max()),
                "mean": float(raw_mfcc.mean()),
                "std": float(raw_mfcc.std()),
            }
        mfcc_meta_path.write_text(json.dumps(mfcc_metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    def _describe_imu(self, topic: str, raw_count: int, data: ImuData) -> str:
        window = np.asarray(
            [[sample.gx, sample.gy, sample.gz, sample.ax, sample.ay, sample.az] for sample in data.samples],
            dtype=np.float32,
        )
        normalized = normalize_window(window, self.gesture_mean, self.gesture_std)
        base = f"[DEBUG][IMU][RESAMPLED] topic={topic} count={raw_count}->{len(data.samples)} {self._format_stats(window)}"
        debug = self.imu_preprocessor.last_debug or {}
        raw_stats = debug.get("raw")
        out_stats = debug.get("out")
        if not isinstance(raw_stats, dict) or not isinstance(out_stats, dict):
            return base
        return (
            base
            + "\n"
            + f"[DEBUG][IMU][RAW] raw_count={debug['raw_count']} {self._format_debug_stats(raw_stats)} "
              f"first={debug['raw_first']} last={debug['raw_last']}"
            + "\n"
            + f"[DEBUG][IMU][RESAMPLED] out_count={debug['out_count']} {self._format_debug_stats(out_stats)} "
              f"first={debug['out_first']} last={debug['out_last']}"
            + "\n"
            + f"[DEBUG][IMU][MODEL_INPUT] normalized {self._format_stats(normalized)}"
        )

    def _describe_voice(self, topic: str, data: VoiceMfccData) -> str:
        mfcc = np.asarray(data.features, dtype=np.float32)
        meta = self._last_voice_debug_meta or "source=unknown"
        base = f"[DEBUG][VOICE][MFCC] topic={topic} {meta} {self._format_stats(mfcc)}"
        feature_stats = self._last_voice_feature_stats or {}
        debug = self.voice_preprocessor.last_debug or {}
        raw_stats = debug.get("raw")
        trimmed_stats = debug.get("trimmed")
        normalized_stats = debug.get("normalized")
        emphasized_stats = debug.get("emphasized")
        if not all(isinstance(item, dict) for item in (raw_stats, trimmed_stats, normalized_stats, emphasized_stats)):
            return base
        return (
            base
            + "\n"
            + f"[DEBUG][VOICE][RAW_AUDIO] raw_samples={debug['raw_samples']} trimmed_samples={debug['trimmed_samples']} "
              f"normalized_samples={debug['normalized_samples']} emphasized_samples={debug['emphasized_samples']} "
              f"pre_emphasis={debug['pre_emphasis']:.4f}"
            + "\n"
            + f"[DEBUG][VOICE][AUGMENTED_AUDIO] raw_rms={raw_stats['rms']:.4f} trimmed_rms={trimmed_stats['rms']:.4f} "
              f"normalized_rms={normalized_stats['rms']:.4f} emphasized_rms={emphasized_stats['rms']:.4f} "
              f"raw_peak={raw_stats['peak']:.4f} normalized_peak={normalized_stats['peak']:.4f}"
            + "\n"
            + f"[DEBUG][VOICE][MFCC] power_shape={debug['power_shape']} mel_shape={debug['mel_shape']} "
              f"mfcc_shape={debug['mfcc_shape']}"
            + "\n"
            + f"[DEBUG][VOICE][MODEL_INPUT] raw_mfcc_mean={feature_stats.get('raw_mean', 0.0):.4f} "
              f"raw_mfcc_std={feature_stats.get('raw_std', 0.0):.4f} "
              f"normalized_mfcc_mean={feature_stats.get('norm_mean', 0.0):.4f} "
              f"normalized_mfcc_std={feature_stats.get('norm_std', 0.0):.4f}"
        )

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

    def _parse_voice_payload(self, topic: str, device_id: Optional[str], payload: bytes) -> Optional[VoiceMfccData]:
        parsed = try_parse_json(payload)
        if isinstance(parsed, dict):
            packet_type = parsed.get("type")
            packet_data = parsed.get("data")
            if packet_type == MessageKind.VOICE_MFCC.value and isinstance(packet_data, dict):
                return VoiceMfccData.from_dict(packet_data)

            candidate = packet_data if isinstance(packet_data, dict) else parsed
            if "features" in candidate:
                return VoiceMfccData.from_dict(candidate)

            if self.voice_reconstructor.is_chunk_payload(candidate):
                chunk = self.voice_reconstructor.add_chunk(candidate)
                print(f"Chunk {chunk.received_chunks}/{chunk.total_chunks} for {chunk.message_id}")
                if not chunk.is_complete:
                    return None
                print(f"Complete audio received: {chunk.message_id}")
                return self._mfcc_from_file_bytes(topic, device_id, chunk.audio_bytes, chunk.suffix)

            audio_bytes = extract_audio_bytes(candidate)
            if audio_bytes is not None:
                filename = str(candidate.get("filename") or candidate.get("name") or "")
                suffix = Path(filename).suffix or self.args.voice_file_extension
                return self._mfcc_from_file_bytes(topic, device_id, audio_bytes, suffix)

        return self._mfcc_from_file_bytes(topic, device_id, payload, self.args.voice_file_extension)

    def _mfcc_from_file_bytes(self, topic: str, device_id: Optional[str], payload: bytes, suffix: str) -> VoiceMfccData:
        temp_path = self.spool_dir / f"voice-{uuid.uuid4().hex}{normalize_suffix(suffix)}"
        self._last_voice_debug_meta = f"suffix={normalize_suffix(suffix)} bytes={len(payload)}"
        temp_path.write_bytes(payload)
        try:
            raw_mfcc = m4a_to_mfcc_matrix(
                temp_path,
                sample_rate=self.args.voice_sample_rate,
                ffmpeg_path=self.args.ffmpeg_path,
                preprocessor=self.voice_preprocessor,
            )
            mfcc = normalize_feature_matrix(raw_mfcc, self.voice_mean, self.voice_std)
            self._last_voice_feature_stats = {
                "raw_mean": float(raw_mfcc.mean()),
                "raw_std": float(raw_mfcc.std()),
                "norm_mean": float(mfcc.mean()),
                "norm_std": float(mfcc.std()),
            }
            self._capture_voice_audio(topic, device_id, payload, suffix, mfcc, raw_mfcc=raw_mfcc)
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
        device_id: Optional[str],
        source_topic: str,
    ) -> None:
        packet = Packet(kind=kind, data=result.to_dict()).to_dict()
        if device_id is not None:
            packet["device_id"] = device_id
            packet["player"] = device_id
        packet["source_topic"] = source_topic
        self.client.publish(topic, json.dumps(packet, separators=(",", ":"), ensure_ascii=True), qos=self.args.publish_qos)

    def _publish_error(self, source_topic: str, message: str, device_id: Optional[str]) -> None:
        if not self.args.error_topic:
            return
        payload = {"type": "error", "data": {"source_topic": source_topic, "message": message}}
        if device_id is not None:
            payload["device_id"] = device_id
            payload["player"] = device_id
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
    parser.add_argument("--imu-max-count", type=int, default=300)
    parser.add_argument("--voice-topic", default=DEFAULT_VOICE_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--pokemon-topic", default=DEFAULT_POKEMON_TOPIC)
    parser.add_argument("--error-topic", default=DEFAULT_ERROR_TOPIC)
    parser.add_argument("--cafile", default=DEFAULT_CAFILE)
    parser.add_argument("--certfile", default=None)
    parser.add_argument("--keyfile", default=None)
    parser.add_argument("--tls-insecure", action="store_true")
    parser.add_argument("--voice-file-extension", default=".m4a")
    parser.add_argument("--voice-chunk-extension", default=".wav")
    parser.add_argument("--voice-chunk-timeout-s", type=float, default=30.0)
    parser.add_argument("--voice-sample-rate", type=int, default=16000)
    parser.add_argument("--gesture-mean", default=str(DEFAULT_GESTURE_MEAN_PATH))
    parser.add_argument("--gesture-std", default=str(DEFAULT_GESTURE_STD_PATH))
    parser.add_argument("--voice-mean", default=str(DEFAULT_VOICE_MEAN_PATH))
    parser.add_argument("--voice-std", default=str(DEFAULT_VOICE_STD_PATH))
    parser.add_argument("--voice-spool-dir", default=str(DEFAULT_SPOOL_DIR))
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))
    parser.add_argument("--session-name", default=None)
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print comprehensive payload, preprocessing, and error debug output for IMU and voice.",
    )
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
