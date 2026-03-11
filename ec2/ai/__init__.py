"""Public exports for the AI-side EC2 <-> Ultra96 communication package.

This package is split by responsibility:
- `audio.py`: decode `.m4a` and extract the fixed-size MFCC matrix
- `messages.py`: validate and build the agreed WebSocket packet shapes
- `sender.py`: EC2 -> Ultra96 messages (`imu`, `voice_mfcc`)
- `receiver.py`: Ultra96 -> EC2 messages (`action`, `pokemon`)
- `service.py`: lightweight glue around sender + receiver
"""

from .audio import VoicePreprocessor, decode_m4a_to_waveform, m4a_to_mfcc_matrix
from .messages import (
    ClassificationData,
    ImuData,
    ImuSample,
    MessageKind,
    Packet,
    VoiceMfccData,
    build_imu_data,
    build_voice_mfcc_data,
)
from .receiver import AIReceiver
from .sender import AISender
from .service import AIService

__all__ = [
    "AIReceiver",
    "AIService",
    "AISender",
    "ClassificationData",
    "ImuData",
    "ImuSample",
    "MessageKind",
    "Packet",
    "VoiceMfccData",
    "VoicePreprocessor",
    "build_imu_data",
    "build_voice_mfcc_data",
    "decode_m4a_to_waveform",
    "m4a_to_mfcc_matrix",
]
