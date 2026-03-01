#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


VOICE_NUM_MFCC = 40
VOICE_NUM_FRAMES = 50
VOICE_NUM_CLASSES = 3
VOICE_B1_CH = 16
VOICE_B1_T = 25
VOICE_B2_CH = 32
VOICE_B2_T = 25


def _quant_ap_fixed(x: np.ndarray | float, total_bits: int, int_bits: int) -> np.ndarray:
    """Approximate ap_fixed<..., AP_TRN, AP_SAT>."""
    frac_bits = total_bits - int_bits
    scale = float(1 << frac_bits)
    min_v = -(1 << (int_bits - 1))
    max_v = (1 << (int_bits - 1)) - (1.0 / scale)
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, min_v, max_v)
    # AP_TRN approximation: truncation toward zero.
    arr = np.trunc(arr * scale) / scale
    return arr.astype(np.float32)


def to_data(x: np.ndarray | float) -> np.ndarray:
    return _quant_ap_fixed(x, total_bits=16, int_bits=8)


def to_acc(x: np.ndarray | float) -> np.ndarray:
    return _quant_ap_fixed(x, total_bits=48, int_bits=20)


def relu_data(x: np.ndarray | float) -> np.ndarray:
    y = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    return to_data(y)


def q88_from_axis_word(word_u32: int) -> float:
    raw = int(word_u32) & 0xFFFF
    if raw >= 0x8000:
        raw -= 0x10000
    # Bit-cast in C++ preserves Q8.8 encoded value.
    return float(to_data(raw / 256.0))


def q88_pack_words_from_float(flat: np.ndarray) -> np.ndarray:
    q = np.round(np.clip(flat.astype(np.float32), -128.0, 127.99609375) * 256.0).astype(np.int32)
    return (q & 0xFFFF).astype(np.uint32)


def parse_weights_header(weights_path: Path) -> Dict[str, np.ndarray]:
    txt = weights_path.read_text(encoding="utf-8")
    names = ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b")
    out: Dict[str, np.ndarray] = {}
    for name in names:
        m = re.search(rf"static const data_t {name}\[\d+\]\s*=\s*\{{(.*?)\}};", txt, flags=re.S)
        if not m:
            raise ValueError(f"Could not parse array '{name}' from {weights_path}")
        nums = [s.strip() for s in m.group(1).replace("\n", " ").split(",") if s.strip()]
        vals = np.array([float(v) for v in nums], dtype=np.float32)
        # data_t in C++ is ap_fixed<16,8>; quantize once at load.
        out[name] = to_data(vals)
    return out


def parse_tb_header(tb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    txt = tb_path.read_text(encoding="utf-8")
    m_x = re.search(r"voice_tb_input_q88\[.*?\]\s*=\s*\{(.*?)\};", txt, flags=re.S)
    m_y = re.search(r"voice_tb_expected\[.*?\]\s*=\s*\{(.*?)\};", txt, flags=re.S)
    if not (m_x and m_y):
        raise ValueError(f"Could not parse testbench arrays from {tb_path}")

    rows = re.findall(r"\{([^{}]+)\}", m_x.group(1), flags=re.S)
    x = np.array(
        [[int(v.strip()) for v in row.split(",") if v.strip()] for row in rows],
        dtype=np.uint32,
    )
    y = np.array([int(v.strip()) for v in m_y.group(1).split(",") if v.strip()], dtype=np.int64)
    return x, y


def voice_cnn_cpp_style(words_tc_q88: np.ndarray, w: Dict[str, np.ndarray]) -> int:
    """
    Reference translation of hls/voice_cnn.cpp.
    Input words must be length 40*50 in stream order [t][c], packed Q8.8 in low 16 bits.
    """
    if words_tc_q88.shape[0] != VOICE_NUM_FRAMES * VOICE_NUM_MFCC:
        raise ValueError(f"Expected {VOICE_NUM_FRAMES * VOICE_NUM_MFCC} words, got {words_tc_q88.shape[0]}")

    conv1_w = w["conv1_w"]
    conv1_b = w["conv1_b"]
    conv2_w = w["conv2_w"]
    conv2_b = w["conv2_b"]
    fc_w = w["fc_w"]
    fc_b = w["fc_b"]

    input_pad = np.zeros((VOICE_NUM_MFCC, VOICE_NUM_FRAMES + 2), dtype=np.float32)
    b1_out = np.zeros((VOICE_B1_CH, VOICE_B1_T), dtype=np.float32)
    b1_pad = np.zeros((VOICE_B1_CH, VOICE_B1_T + 2), dtype=np.float32)
    b2_out = np.zeros((VOICE_B2_CH, VOICE_B2_T), dtype=np.float32)

    # 1) Read input stream [t][c] into padded buffer [c][t+1].
    idx = 0
    for t in range(VOICE_NUM_FRAMES):
        for c in range(VOICE_NUM_MFCC):
            input_pad[c, t + 1] = q88_from_axis_word(int(words_tc_q88[idx]))
            idx += 1
    input_pad = to_data(input_pad)

    # 2) Block1 conv+relu+maxpool2.
    for o in range(VOICE_B1_CH):
        for t in range(VOICE_B1_T):
            max_val = to_data(-128.0).item()
            for p in range(2):
                curr_t = t * 2 + p
                pad_t = curr_t + 1

                s = float(to_acc(conv1_b[o]))
                for i in range(VOICE_NUM_MFCC):
                    base = o * (VOICE_NUM_MFCC * 3) + i * 3
                    for k in range(3):
                        mul = float(to_acc(float(input_pad[i, pad_t + (k - 1)]) * float(conv1_w[base + k])))
                        s = float(to_acc(s + mul))

                v = float(relu_data(s))
                if v > max_val:
                    max_val = v
            b1_out[o, t] = max_val
    b1_out = to_data(b1_out)

    # 3) Build padded b1.
    b1_pad[:, 1 : 1 + VOICE_B1_T] = b1_out
    b1_pad = to_data(b1_pad)

    # 4) Block2 conv+relu.
    for o in range(VOICE_B2_CH):
        for t in range(VOICE_B2_T):
            pad_t = t + 1
            s = float(to_acc(conv2_b[o]))
            for i in range(VOICE_B1_CH):
                base = o * (VOICE_B1_CH * 3) + i * 3
                for k in range(3):
                    mul = float(to_acc(float(b1_pad[i, pad_t + (k - 1)]) * float(conv2_w[base + k])))
                    s = float(to_acc(s + mul))
            b2_out[o, t] = float(relu_data(s))
    b2_out = to_data(b2_out)

    # 5) Global average pool.
    pooled = np.zeros((VOICE_B2_CH,), dtype=np.float32)
    inv_t = float(to_data(1.0 / VOICE_B2_T))
    for c in range(VOICE_B2_CH):
        s = float(to_acc(0.0))
        for t in range(VOICE_B2_T):
            s = float(to_acc(s + float(b2_out[c, t])))
        pooled[c] = float(to_data(float(to_acc(s * inv_t))))
    pooled = to_data(pooled)

    # 6) FC.
    logits = np.zeros((VOICE_NUM_CLASSES,), dtype=np.float32)
    for c in range(VOICE_NUM_CLASSES):
        s = float(to_acc(fc_b[c]))
        base = c * VOICE_B2_CH
        for i in range(VOICE_B2_CH):
            mul = float(to_acc(float(pooled[i]) * float(fc_w[base + i])))
            s = float(to_acc(s + mul))
        logits[c] = float(to_data(s))
    logits = to_data(logits)

    # 7) Argmax.
    best = 0
    best_score = float(logits[0])
    for c in range(1, VOICE_NUM_CLASSES):
        if float(logits[c]) > best_score:
            best_score = float(logits[c])
            best = c
    return int(best)


def run_tb(weights_path: Path, tb_path: Path, limit: int = 0) -> None:
    w = parse_weights_header(weights_path)
    x, y = parse_tb_header(tb_path)

    n = len(y) if limit <= 0 else min(limit, len(y))
    correct = 0
    preds: List[int] = []
    for i in range(n):
        p = voice_cnn_cpp_style(x[i], w)
        preds.append(p)
        if p == int(y[i]):
            correct += 1

    acc = 100.0 * correct / max(n, 1)
    print(f"TB cases: {n}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"First 20 preds:   {preds[:20]}")
    print(f"First 20 labels:  {y[:20].tolist()}")


def run_npy(weights_path: Path, features_path: Path, labels_path: Path, limit: int = 0) -> None:
    w = parse_weights_header(weights_path)
    x = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64).reshape(-1)

    if x.ndim != 3 or x.shape[1:] != (40, 50):
        raise ValueError(f"Expected features [N,40,50], got {x.shape}")

    n = min(len(x), len(y))
    if limit > 0:
        n = min(n, limit)

    correct = 0
    preds: List[int] = []
    used = 0
    for i in range(n):
        yt = int(y[i])
        if yt < 0 or yt >= 3:
            continue
        # Match PS/HLS path: flatten [C,T] to stream order [t][c], then Q8.8 pack.
        flat_tc = x[i].T.reshape(-1)
        words = q88_pack_words_from_float(flat_tc)
        p = voice_cnn_cpp_style(words, w)
        preds.append(p)
        used += 1
        if p == yt:
            correct += 1

    acc = 100.0 * correct / max(used, 1)
    print(f"NPY samples used: {used}")
    print(f"Correct:          {correct}")
    print(f"Accuracy:         {acc:.2f}%")
    print(f"First 20 preds:   {preds[:20]}")
    print(f"First 20 labels:  {[int(v) for v in y[:20]]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Python reference of hls/voice_cnn.cpp (non-HLS).")
    ap.add_argument("--weights", default="hls/voice_cnn_weights.h", help="Path to voice_cnn_weights.h")
    ap.add_argument("--tb", default="hls/voice_tb_cases.h", help="Path to voice_tb_cases.h")
    ap.add_argument("--features", default="", help="Path to voice_X_test.npy")
    ap.add_argument("--labels", default="", help="Path to voice_y_test.npy")
    ap.add_argument("--limit", type=int, default=0, help="Limit samples/cases (0 = all)")
    ap.add_argument("--mode", choices=["tb", "npy"], default="tb")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if args.mode == "tb":
        run_tb(weights_path, Path(args.tb), limit=args.limit)
    else:
        if not args.features or not args.labels:
            raise ValueError("--features and --labels are required in --mode npy")
        run_npy(weights_path, Path(args.features), Path(args.labels), limit=args.limit)


if __name__ == "__main__":
    main()
