#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def q88_pack_u32(x: np.ndarray) -> np.ndarray:
    q = np.round(np.clip(x, -128.0, 127.99609375) * 256.0).astype(np.int32)
    return (q & 0xFFFF).astype(np.uint32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate HLS gesture testbench dataset header from npy files.")
    ap.add_argument("--features", required=True, help="Path to gesture_X_test.npy")
    ap.add_argument("--labels", required=True, help="Path to gesture_y_test.npy")
    ap.add_argument("--out", default="hls/gesture/gesture_tb_cases.h", help="Output header path")
    ap.add_argument("--num-cases", type=int, default=300, help="Number of random test samples to export")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic sampling")
    args = ap.parse_args()

    x = np.load(args.features).astype(np.float32)
    y = np.load(args.labels).astype(np.int64).reshape(-1)

    if x.ndim != 3:
        raise ValueError(f"Expected gesture features rank-3, got {x.shape}")

    if x.shape[1:] == (60, 6):
        xw = x
    elif x.shape[1:] == (6, 60):
        xw = np.transpose(x, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Expected gesture shape [N,60,6] or [N,6,60], got {x.shape}")

    n = min(len(xw), len(y))
    xw = xw[:n]
    y = y[:n]

    keep = (y >= 0) & (y < 6)
    xw = xw[keep]
    y = y[keep]
    if len(xw) == 0:
        raise RuntimeError("No valid gesture samples found after label filtering.")

    num_cases = min(args.num_cases, len(xw))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(xw), size=num_cases, replace=False)

    xs = xw[idx]
    ys = y[idx]

    xs_flat = xs.reshape(num_cases, -1).astype(np.float32)
    xs_q88 = q88_pack_u32(xs_flat)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("#ifndef GESTURE_TB_CASES_H\n")
        f.write("#define GESTURE_TB_CASES_H\n\n")
        f.write("#include <cstdint>\n")
        f.write('#include "gesture_typedefs.h"\n\n')
        f.write(f"#define GESTURE_TB_NUM_CASES {num_cases}\n")
        f.write("#define GESTURE_TB_WORDS (WINDOW_SIZE * NUM_SENSORS)\n\n")

        f.write("static const uint32_t gesture_tb_input_q88[GESTURE_TB_NUM_CASES][GESTURE_TB_WORDS] = {\n")
        for i in range(num_cases):
            row = ", ".join(str(int(v)) for v in xs_q88[i].tolist())
            trailing = "," if i < num_cases - 1 else ""
            f.write(f"    {{{row}}}{trailing}\n")
        f.write("};\n\n")

        f.write("static const int gesture_tb_expected[GESTURE_TB_NUM_CASES] = {")
        f.write(", ".join(str(int(v)) for v in ys.tolist()))
        f.write("};\n\n")
        f.write("#endif\n")

    vals, cts = np.unique(ys, return_counts=True)
    dist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cts))
    print(f"Wrote {out} with {num_cases} cases. Label distribution: {dist}")


if __name__ == "__main__":
    main()
