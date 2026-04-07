from __future__ import annotations

"""Integration-local Ultra96 hardware constants and DMA helpers."""

import time

import numpy as np

DEFAULT_XSA = "dual_cnn.xsa"

GESTURE_CORE_NAME = "gesture_cnn_0"
VOICE_CORE_NAME = "voice_cnn_0"
GESTURE_DMA_NAME = "axi_dma_1"
VOICE_DMA_NAME = "axi_dma_0"

GESTURE_LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
VOICE_LABELS = ["Bulbasaur", "Charizard", "Greninja", "Lugia", "Mewtwo", "Pikachu"]

# Current class meanings used by the latest datasets/model wiring:
# Gesture: 0=Raise, 1=Shake, 2=Chop, 3=Stir, 4=Swing, 5=Punch
# Voice: 0=bulbasaur, 1=charizard, 2=greninja, 3=lugia, 4=mewtwo, 5=pikachu


def q88_pack_u32(x: np.ndarray) -> np.ndarray:
    """Pack float values to signed Q8.8 in AXIS data[15:0] (uint32 words)."""

    q = np.round(np.clip(x, -128.0, 127.99609375) * 256.0).astype(np.int32)
    return (q & 0xFFFF).astype(np.uint32)


def reset_dma(dma) -> None:
    for ch in [dma.sendchannel, dma.recvchannel]:
        ch._mmio.write(ch._offset + 0x00, 0x4)
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        ch._mmio.write(ch._offset + 0x00, 0x1)
    time.sleep(0.01)


def run_dma(dma, in_buf, out_buf, timeout_s: float) -> None:
    in_buf.flush()
    out_buf.invalidate()

    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    t_wait0 = time.perf_counter()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.perf_counter() - t_wait0 > timeout_s:
            raise TimeoutError("DMA timeout")

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    out_buf.invalidate()


def stop_core(core) -> None:
    core.write(0x00, 0x00)


def start_core(core) -> None:
    core.write(0x00, 0x01)
