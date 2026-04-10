#!/usr/bin/env python3
"""Build a dated gesture dataset by appending SQLite-captured windows to a base CSV."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from gesture_feature_pipeline import build_dashboard_gesture_dataframe, discover_dashboard_gesture_roots


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
LABEL_TOKEN_TO_ID = {
    "raise": 0,
    "shake": 1,
    "chop": 2,
    "stir": 3,
    "swing": 4,
    "punch": 5,
}


@dataclass
class ImportSpec:
    label_token: str
    db_path: str
    exclude_action_ids: list[int]


def parse_db_spec(raw: str) -> ImportSpec:
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    payload: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid --db-spec segment: {part!r}")
        key, value = part.split("=", 1)
        payload[key.strip()] = value.strip()

    label_token = payload.get("label", "").lower()
    if label_token not in LABEL_TOKEN_TO_ID:
        raise ValueError(f"Unsupported label token in --db-spec: {label_token!r}")
    db_path = payload.get("path")
    if not db_path:
        raise ValueError("Missing path=... in --db-spec")
    exclude_raw = payload.get("exclude", "")
    exclude_ids = [int(token) for token in exclude_raw.split(",") if token.strip()]
    return ImportSpec(label_token=label_token, db_path=db_path, exclude_action_ids=exclude_ids)


def load_base_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Base CSV is missing columns: {missing}")
    return df[CSV_COLUMNS].copy()


def iter_action_ids(conn: sqlite3.Connection) -> Iterable[tuple[int, str]]:
    rows = conn.execute("SELECT id, action_name FROM actions ORDER BY id").fetchall()
    for action_id, action_name in rows:
        yield int(action_id), str(action_name or "")


def load_action_window(conn: sqlite3.Connection, action_id: int) -> list[tuple[float, float, float, float, float, float]]:
    rows = conn.execute(
        """
        SELECT gx, gy, gz, ax, ay, az
        FROM action_samples
        WHERE action_id = ?
        ORDER BY id
        """,
        (action_id,),
    ).fetchall()
    return [(float(gx), float(gy), float(gz), float(ax), float(ay), float(az)) for gx, gy, gz, ax, ay, az in rows]


def append_sqlite_samples(base_df: pd.DataFrame, spec: ImportSpec) -> tuple[pd.DataFrame, dict[str, object]]:
    db_path = Path(spec.db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    label_id = LABEL_TOKEN_TO_ID[spec.label_token]
    label_name = PROJECT_LABEL_ORDER[label_id]
    next_measurement_id = int(base_df["measurement_id"].max()) + 1 if not base_df.empty else 0
    rows_to_append: list[dict[str, object]] = []
    kept_action_ids: list[int] = []
    skipped_short: list[int] = []

    with sqlite3.connect(str(db_path)) as conn:
        for action_id, action_name in iter_action_ids(conn):
            if action_id in spec.exclude_action_ids:
                continue
            window = load_action_window(conn, action_id)
            if len(window) < 10:
                skipped_short.append(action_id)
                continue
            for sequence_id, (gx, gy, gz, ax, ay, az) in enumerate(window):
                rows_to_append.append(
                    {
                        "measurement_id": next_measurement_id,
                        "sequence_id": sequence_id,
                        "label_id": label_id,
                        "label": label_name,
                        "gyro_x": gx,
                        "gyro_y": gy,
                        "gyro_z": gz,
                        "acc_x": ax,
                        "acc_y": ay,
                        "acc_z": az,
                    }
                )
            kept_action_ids.append(action_id)
            next_measurement_id += 1

    appended_df = pd.DataFrame(rows_to_append, columns=CSV_COLUMNS)
    merged_df = pd.concat([base_df, appended_df], ignore_index=True)
    manifest = {
        "label": label_name,
        "db_path": str(db_path),
        "excluded_action_ids": spec.exclude_action_ids,
        "kept_action_ids": kept_action_ids,
        "skipped_short_action_ids": skipped_short,
        "appended_measurements": len(kept_action_ids),
        "appended_rows": int(len(appended_df)),
    }
    return merged_df, manifest


def build_combined_dataframe(base_df: pd.DataFrame, dataset_dir: Path, dashboard_data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    base_combined = base_df.copy()
    base_combined["source"] = dataset_dir.name
    base_combined["path"] = f"../data/gesture/{dataset_dir.name}/imudata.csv"

    dashboard_sources = discover_dashboard_gesture_roots(
        dashboard_data_root,
        required_class_names={"raise", "shake", "chop", "stir", "swing", "punch"},
    )
    if dashboard_sources:
        dashboard_df = build_dashboard_gesture_dataframe(
            dashboard_sources,
            measurement_id_start=int(base_df["measurement_id"].max()) + 1 if not base_df.empty else 0,
        )
    else:
        dashboard_df = pd.DataFrame(columns=list(base_combined.columns))

    combined_df = pd.concat([base_combined, dashboard_df], ignore_index=True)
    manifest = {
        "base_measurements": int(base_df["measurement_id"].nunique()),
        "base_rows": int(len(base_df)),
        "dashboard_sources": [str(path) for path in dashboard_sources],
        "dashboard_measurements": int(dashboard_df["measurement_id"].nunique()) if not dashboard_df.empty else 0,
        "dashboard_rows": int(len(dashboard_df)),
        "dashboard_class_counts": (
            dashboard_df["label"].value_counts().sort_index().to_dict() if not dashboard_df.empty else {}
        ),
        "combined_measurements": int(combined_df["measurement_id"].nunique()),
        "combined_rows": int(len(combined_df)),
    }
    return combined_df, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a new gesture dataset folder by importing SQLite gesture recordings.")
    parser.add_argument("--base-csv", type=Path, required=True, help="Existing canonical imudata.csv to extend.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New dated gesture dataset folder.")
    parser.add_argument("--dashboard-data-root", type=Path, required=True, help="Dashboard data root used to rebuild imudata_combined.csv.")
    parser.add_argument(
        "--db-spec",
        action="append",
        default=[],
        help="Import spec in the form label=<raise|shake|chop|stir|swing|punch>;path=/abs/file.db;exclude=1,2,3",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_df = load_base_dataframe(args.base_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import_specs = [parse_db_spec(raw) for raw in args.db_spec]
    import_manifest = {
        "base_csv": str(args.base_csv.resolve()),
        "output_dir": str(output_dir),
        "imports": [],
    }

    working_df = base_df
    for spec in import_specs:
        working_df, manifest = append_sqlite_samples(working_df, spec)
        import_manifest["imports"].append(manifest)

    imudata_csv = output_dir / "imudata.csv"
    working_df.to_csv(imudata_csv, index=False)

    combined_df, merge_manifest = build_combined_dataframe(
        working_df,
        dataset_dir=output_dir,
        dashboard_data_root=args.dashboard_data_root.resolve(),
    )
    combined_df.to_csv(output_dir / "imudata_combined.csv", index=False)
    (output_dir / "dashboard_merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2), encoding="utf-8")
    (output_dir / "sqlite_import_manifest.json").write_text(json.dumps(import_manifest, indent=2), encoding="utf-8")

    print(f"Wrote {imudata_csv} with {working_df['measurement_id'].nunique()} measurements and {len(working_df)} rows")
    print(f"Wrote {output_dir / 'imudata_combined.csv'} with {combined_df['measurement_id'].nunique()} measurements and {len(combined_df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
