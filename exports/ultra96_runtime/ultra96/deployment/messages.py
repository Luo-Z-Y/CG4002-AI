from __future__ import annotations

"""Packet definitions for the Ultra96 <-> EC2 WebSocket link."""

from dataclasses import dataclass as _dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Union


def dataclass(*args, **kwargs):
    """Backwards-compatible wrapper for Python < 3.10 where `slots` is unsupported."""

    try:
        return _dataclass(*args, **kwargs)
    except TypeError:
        kwargs.pop("slots", None)
        return _dataclass(*args, **kwargs)


class MessageKind(str, Enum):
    IMU = "imu"
    VOICE_MFCC = "voice_mfcc"
    ACTION = "action"
    POKEMON = "pokemon"


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a number")
    return float(value)


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _require_number_row(value: Any, row_idx: int, expected_cols: int) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"features[{row_idx}] must be an array")
    if len(value) != expected_cols:
        raise ValueError(f"features[{row_idx}] must contain exactly {expected_cols} numbers")
    return [_require_number(item, f"features[{row_idx}][{col_idx}]") for col_idx, item in enumerate(value)]


@dataclass(slots=True)
class ImuSample:
    """Single IMU reading with raw gyroscope and accelerometer axes.

    Field order matches the training CSV: gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z.
    Prior builds used y/p/r (yaw/pitch/roll); from 2026-03-28 the pipeline uses raw gyro.
    """

    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "gz": self.gz,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImuSample":
        data = _require_mapping(data, "imu_sample")
        return cls(
            gx=_require_number(data.get("gx"), "imu_sample.gx"),
            gy=_require_number(data.get("gy"), "imu_sample.gy"),
            gz=_require_number(data.get("gz"), "imu_sample.gz"),
            ax=_require_number(data.get("ax"), "imu_sample.ax"),
            ay=_require_number(data.get("ay"), "imu_sample.ay"),
            az=_require_number(data.get("az"), "imu_sample.az"),
        )


@dataclass(slots=True)
class ImuData:
    samples: list[ImuSample]
    count: Optional[int] = None

    def __post_init__(self) -> None:
        actual_count = len(self.samples)
        if actual_count == 0:
            raise ValueError("imu.data.samples must be a non-empty array")
        if self.count is None:
            self.count = actual_count
        if self.count != actual_count:
            raise ValueError(f"imu.data.count={self.count} does not match len(samples)={actual_count}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImuData":
        data = _require_mapping(data, "imu.data")
        raw_samples = data.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("imu.data.samples must be a non-empty array")
        count = _require_int(data.get("count"), "imu.data.count")
        return cls(
            samples=[ImuSample.from_dict(sample) for sample in raw_samples],
            count=count,
        )


@dataclass(slots=True)
class VoiceMfccData:
    features: list[list[float]]
    shape: tuple[int, int] = (40, 50)

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if (rows, cols) != (40, 50):
            raise ValueError(f"voice_mfcc.data.shape must be [40, 50], got [{rows}, {cols}]")
        if len(self.features) != rows:
            raise ValueError(f"voice_mfcc.data.features must contain exactly {rows} rows")
        for row_idx, row in enumerate(self.features):
            if len(row) != cols:
                raise ValueError(f"voice_mfcc.data.features[{row_idx}] must contain exactly {cols} values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": [self.shape[0], self.shape[1]],
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VoiceMfccData":
        data = _require_mapping(data, "voice_mfcc.data")
        raw_shape = data.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            raise ValueError("voice_mfcc.data.shape must be a 2-element array")
        rows = _require_int(raw_shape[0], "voice_mfcc.data.shape[0]")
        cols = _require_int(raw_shape[1], "voice_mfcc.data.shape[1]")
        raw_features = data.get("features")
        if not isinstance(raw_features, list):
            raise TypeError("voice_mfcc.data.features must be an array")
        features = [_require_number_row(row, row_idx, cols) for row_idx, row in enumerate(raw_features)]
        return cls(features=features, shape=(rows, cols))


@dataclass(slots=True)
class ClassificationData:
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], message_kind: MessageKind) -> "ClassificationData":
        data = _require_mapping(data, f"{message_kind.value}.data")
        return cls(
            label=_require_string(data.get("label"), f"{message_kind.value}.data.label"),
            confidence=_require_number(data.get("confidence"), f"{message_kind.value}.data.confidence"),
        )


_PAYLOAD_TYPES = {
    MessageKind.IMU: ImuData,
    MessageKind.VOICE_MFCC: VoiceMfccData,
    MessageKind.ACTION: ClassificationData,
    MessageKind.POKEMON: ClassificationData,
}


@dataclass(slots=True)
class Packet:
    kind: MessageKind
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Packet":
        value = _require_mapping(value, "packet")
        kind = MessageKind(_require_string(value.get("type"), "packet.type"))
        payload_type = _PAYLOAD_TYPES[kind]
        raw_data = _require_mapping(value.get("data"), "packet.data")
        if kind in (MessageKind.ACTION, MessageKind.POKEMON):
            data = payload_type.from_dict(raw_data, kind).to_dict()
        else:
            data = payload_type.from_dict(raw_data).to_dict()
        return cls(kind=kind, data=data)

    def decode_data(self) -> Union["ImuData", "VoiceMfccData", "ClassificationData"]:
        payload_type = _PAYLOAD_TYPES[self.kind]
        if self.kind in (MessageKind.ACTION, MessageKind.POKEMON):
            return payload_type.from_dict(self.data, self.kind)
        return payload_type.from_dict(self.data)


def build_imu_data(samples: Sequence[Union[ImuSample, Mapping[str, Any]]]) -> ImuData:
    normalized_samples = [
        sample if isinstance(sample, ImuSample) else ImuSample.from_dict(sample)
        for sample in samples
    ]
    return ImuData(samples=normalized_samples)


def build_voice_mfcc_data(features: Sequence[Sequence[float]]) -> VoiceMfccData:
    normalized_rows = [
        [_require_number(item, f"voice_mfcc.features[{row_idx}][{col_idx}]") for col_idx, item in enumerate(row)]
        for row_idx, row in enumerate(features)
    ]
    return VoiceMfccData(features=normalized_rows)
