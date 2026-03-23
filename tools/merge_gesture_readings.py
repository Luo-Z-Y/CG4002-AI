from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


PACKET_MARKER = "==== Gesture Packet ===="
CANONICAL_ORDER = [
    "pledge",
    "shake_fist",
    "vertical_chop",
    "circular_stir",
    "horizontal_swing",
    "punch",
]


@dataclass(frozen=True)
class SourceBlock:
    source_file: str
    source_index: int
    text: str


def normalize_label(stem: str) -> str | None:
    token = stem.lower().strip()
    token = re.sub(r"^\d+_?", "", token)
    token = token.replace("-", "_").replace(" ", "_")
    aliases = {
        "pledge": "pledge",
        "raise": "pledge",
        "shake_fist": "shake_fist",
        "shake3": "shake_fist",
        "vertical_chop": "vertical_chop",
        "vetical_chop": "vertical_chop",
        "vertical": "vertical_chop",
        "circular": "circular_stir",
        "circular_stir": "circular_stir",
        "circular_stir_clockwise": "circular_stir",
        "horizontal": "horizontal_swing",
        "horizontal_swing": "horizontal_swing",
        "punch": "punch",
    }
    return aliases.get(token)


def split_packets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = text.split(PACKET_MARKER)
    packets: list[str] = []
    for part in parts[1:]:
        block = f"{PACKET_MARKER}{part}".strip()
        if block:
            packets.append(block)
    return packets


def collect_blocks(folder: Path) -> dict[str, list[SourceBlock]]:
    grouped = {label: [] for label in CANONICAL_ORDER}
    for src in sorted(folder.glob("*.txt")):
        label = normalize_label(src.stem)
        if label is None:
            continue
        for idx, packet in enumerate(split_packets(src), start=1):
            grouped[label].append(
                SourceBlock(source_file=str(src.resolve()), source_index=idx, text=packet)
            )
    return grouped


def merge_blocks(
    new_dir: Path,
    old_dirs: list[Path],
    out_dir: Path,
    new_count: int,
    old_count: int,
) -> dict[str, dict[str, object]]:
    new_blocks = collect_blocks(new_dir)
    old_collected = {label: [] for label in CANONICAL_ORDER}
    for old_dir in old_dirs:
        blocks = collect_blocks(old_dir)
        for label in CANONICAL_ORDER:
            old_collected[label].extend(blocks[label])

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}

    for label in CANONICAL_ORDER:
        picked_new = new_blocks[label][:new_count]
        picked_old = old_collected[label][:old_count]
        if len(picked_new) < new_count:
            raise ValueError(
                f"{label}: expected {new_count} new packets in {new_dir}, found {len(picked_new)}"
            )
        if len(picked_old) < old_count:
            raise ValueError(
                f"{label}: expected {old_count} old packets across {old_dirs}, found {len(picked_old)}"
            )

        merged = picked_new + picked_old
        output_path = out_dir / f"{label}.txt"
        output_path.write_text("\n\n".join(block.text for block in merged) + "\n", encoding="utf-8")

        manifest[label] = {
            "output_file": str(output_path.resolve()),
            "new_count": len(picked_new),
            "old_count": len(picked_old),
            "total_count": len(merged),
            "new_sources": [
                {"file": block.source_file, "packet_index": block.source_index} for block in picked_new
            ],
            "old_sources": [
                {"file": block.source_file, "packet_index": block.source_index} for block in picked_old
            ],
        }

    manifest_path = out_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge newest and older raw gesture packets into a balanced combined folder."
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("data/gesture/20260322/cg4002_20_3_readings"),
        help="Folder containing the newest raw gesture logs.",
    )
    parser.add_argument(
        "--old-dir",
        dest="old_dirs",
        type=Path,
        action="append",
        default=[
            Path("data/gesture/20260317/Cg4002 Readings 17_3"),
            Path("data/gesture/20260317/cg4002_14_3_readings"),
        ],
        help="Older raw gesture log folder. May be provided multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/gesture/20260322/cg4002_20_3_readings_new50_old50"),
        help="Destination folder for the merged raw logs.",
    )
    parser.add_argument(
        "--new-count",
        type=int,
        default=50,
        help="How many packets to take per gesture from the newest folder.",
    )
    parser.add_argument(
        "--old-count",
        type=int,
        default=50,
        help="How many packets to take per gesture from the older folders combined.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for path in [args.new_dir, *args.old_dirs]:
        if not path.exists():
            raise FileNotFoundError(f"Missing source folder: {path}")

    manifest = merge_blocks(
        new_dir=args.new_dir,
        old_dirs=args.old_dirs,
        out_dir=args.out_dir,
        new_count=args.new_count,
        old_count=args.old_count,
    )

    print(f"Wrote merged gesture logs to: {args.out_dir.resolve()}")
    for label in CANONICAL_ORDER:
        info = manifest[label]
        print(
            f"{label}: new={info['new_count']} old={info['old_count']} total={info['total_count']}"
        )


if __name__ == "__main__":
    main()
