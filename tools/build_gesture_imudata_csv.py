from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

PACKET_MARKER = "==== Gesture Packet ===="
LEGACY_HEADER_MARKER = "YPR(deg)-X"

CSV_COLUMNS = [
    "measurement_id",
    "sequence_id",
    "label_id",
    "label",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
]

PROJECT_LABEL_ORDER = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
CLASS_MAP = {
    "pledge": 0,
    "raise": 0,
    "shake_fist": 1,
    "shake3": 1,
    "vetical_chop": 2,
    "vertical_chop": 2,
    "vertical": 2,
    "circular": 3,
    "circular_stir": 3,
    "circular_stir_clockwise": 3,
    "cicrular_stir_clockwise": 3,
    "horizontal_swing": 4,
    "horizontal": 4,
    "punch": 5,
}


def normalize_label_token(name: str) -> str:
    token = name.strip().lower().replace(" ", "_").replace("-", "_")
    token = re.sub(r"^\d+_?", "", token)
    return token


def read_log_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    return text.replace("\x00", "").replace("\r\n", "\n").splitlines()


def parse_packet_blocks(path: Path) -> list[np.ndarray]:
    text = "\n".join(read_log_lines(path))
    blocks: list[np.ndarray] = []
    for block in text.split(PACKET_MARKER):
        block = block.strip()
        if not block:
            continue
        rows: list[list[float]] = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("Sample count:") or line.startswith("Idx,"):
                continue
            match = re.match(r"\d+\s*\|\s*(.+)$", line)
            if not match:
                continue
            parts = [part.strip() for part in match.group(1).split("|")]
            if len(parts) != 6:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue
        if rows:
            blocks.append(np.asarray(rows, dtype=np.float32))
    return blocks


def parse_legacy_marker_blocks(path: Path) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    current: list[list[float]] = []
    in_data = False
    for raw_line in read_log_lines(path):
        line = raw_line.strip()
        if LEGACY_HEADER_MARKER in line:
            if current:
                blocks.append(np.asarray(current, dtype=np.float32))
                current = []
            in_data = True
            continue
        if not in_data:
            continue
        if not line or line.startswith("---"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            current.append([float(value) for value in parts])
        except ValueError:
            continue
    if current:
        blocks.append(np.asarray(current, dtype=np.float32))
    return blocks


def parse_plain_window(path: Path) -> list[np.ndarray]:
    for delimiter in (",", None):
        try:
            arr = np.loadtxt(path, dtype=np.float32, delimiter=delimiter)
        except Exception:
            arr = None
        if arr is None:
            continue
        if arr.ndim == 1:
            if arr.size % 6 != 0:
                continue
            arr = arr.reshape(-1, 6)
        if arr.ndim == 2 and arr.shape[1] == 6 and not np.isnan(arr).any():
            return [arr.astype(np.float32)]
    return []


def parse_recordings(path: Path) -> list[np.ndarray]:
    for parser in (parse_packet_blocks, parse_legacy_marker_blocks, parse_plain_window):
        blocks = parser(path)
        if blocks:
            return blocks
    return []


def build_rows(input_dir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    measurement_id = 0

    for txt_path in sorted(input_dir.glob("*.txt")):
        label_token = normalize_label_token(txt_path.stem)
        if label_token not in CLASS_MAP:
            continue

        label_id = CLASS_MAP[label_token]
        label_name = PROJECT_LABEL_ORDER[label_id]
        blocks = parse_recordings(txt_path)

        for block in blocks:
            if len(block) < 10:
                continue
            for sequence_id, sample in enumerate(block):
                rows.append(
                    {
                        "measurement_id": measurement_id,
                        "sequence_id": sequence_id,
                        "label_id": label_id,
                        "label": label_name,
                        "gyro_x": float(sample[0]),
                        "gyro_y": float(sample[1]),
                        "gyro_z": float(sample[2]),
                        "acc_x": float(sample[3]),
                        "acc_y": float(sample[4]),
                        "acc_z": float(sample[5]),
                    }
                )
            measurement_id += 1

    return rows


def write_csv(output_csv: Path, rows: list[dict[str, float | int | str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build imudata.csv from gesture txt logs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing gesture txt files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Target imudata.csv path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the target CSV if it exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV already exists: {args.output_csv}")

    rows = build_rows(args.input_dir)
    if not rows:
        raise RuntimeError(f"No valid gesture recordings found under {args.input_dir}")

    write_csv(args.output_csv, rows)
    measurement_count = len({int(row["measurement_id"]) for row in rows})
    print(f"Wrote {args.output_csv.resolve()} with {measurement_count} measurements and {len(rows)} rows")


if __name__ == "__main__":
    main()
