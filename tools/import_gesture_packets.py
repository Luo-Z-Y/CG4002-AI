from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PACKET_MARKER = "==== Gesture Packet ===="
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


def parse_packets(path: Path) -> list[list[list[float]]]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    packets: list[list[list[float]]] = []

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

            rows.append([float(value) for value in parts])

        if rows:
            packets.append(rows)

    return packets


def next_measurement_id(csv_path: Path) -> int:
    max_measurement_id = -1
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            max_measurement_id = max(max_measurement_id, int(row["measurement_id"]))
    return max_measurement_id + 1


def append_packets(csv_path: Path, packets: list[list[list[float]]], label_id: int, label: str) -> tuple[int, int]:
    start_measurement_id = next_measurement_id(csv_path)
    current_measurement_id = start_measurement_id
    appended_rows = 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        for packet in packets:
            for sequence_id, values in enumerate(packet):
                writer.writerow(
                    {
                        "measurement_id": current_measurement_id,
                        "sequence_id": sequence_id,
                        "label_id": label_id,
                        "label": label,
                        "gyro_x": values[0],
                        "gyro_y": values[1],
                        "gyro_z": values[2],
                        "acc_x": values[3],
                        "acc_y": values[4],
                        "acc_z": values[5],
                    }
                )
                appended_rows += 1
            current_measurement_id += 1

    return start_measurement_id, current_measurement_id - 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append raw gesture packet logs into an existing imudata.csv dataset.")
    parser.add_argument("--csv", type=Path, required=True, help="Target imudata.csv file.")
    parser.add_argument("--input", type=Path, required=True, help="Source raw gesture txt log.")
    parser.add_argument("--label-id", type=int, required=True, help="Numeric label id to assign.")
    parser.add_argument("--label", required=True, help="Label name to assign.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    packets = parse_packets(args.input)
    if not packets:
        raise RuntimeError(f"No gesture packets found in {args.input}")

    start_id, end_id = append_packets(args.csv, packets, label_id=args.label_id, label=args.label)
    print(
        f"Appended {len(packets)} packets from {args.input} to {args.csv} "
        f"as label_id={args.label_id} label={args.label} "
        f"measurement_ids={start_id}-{end_id}"
    )


if __name__ == "__main__":
    main()
