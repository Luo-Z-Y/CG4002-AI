from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CG4002 assessment suite using dual_cnn_test.py and collect outputs."
    )
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")
    p.add_argument("--runner", default="dual_cnn_test.py", help="Path to dual_cnn_test.py.")
    p.add_argument("--xsa-path", default="dual_cnn.xsa")
    p.add_argument("--save-dir", default="../report/evidence_dual")
    p.add_argument("--voice-order", choices=["tc", "ct"], default="tc")
    p.add_argument("--power-w", type=float, default=4.2)
    p.add_argument("--pl-clock-mhz", type=float, default=100.0)
    p.add_argument(
        "--pl-clock-sweep",
        default="75,100,125",
        help="Comma-separated PL clock sweep values in MHz for extended runs.",
    )
    p.add_argument("--cpu-governor", default="performance")
    p.add_argument(
        "--cpu-freq-khz",
        type=int,
        default=1200000,
        help="CPU frequency used for userspace governor run.",
    )
    p.add_argument(
        "--power-sysfs-path",
        default="/sys/class/hwmon/hwmon0/power1_input",
        help="Sysfs path used for power-sensor run.",
    )
    p.add_argument(
        "--power-sysfs-scale",
        type=float,
        default=1e-6,
        help="Scale factor for power sysfs reading (e.g. 1e-6 for microwatts).",
    )
    p.add_argument(
        "--skip-power",
        action="store_true",
        help="Skip the power-profiled dual run.",
    )
    p.add_argument(
        "--tag-prefix",
        default="assess",
        help="Prefix used for run tags.",
    )
    return p.parse_args()


def run_one(
    python_bin: str,
    runner: str,
    save_dir: str,
    xsa_path: str,
    tag: str,
    extra: List[str],
) -> int:
    cmd = [
        python_bin,
        runner,
        "--xsa-path",
        xsa_path,
        "--save-dir",
        save_dir,
        "--tag",
        tag,
    ] + extra
    print("\n[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def newest_summary_dir(save_dir: Path, after_ts: float) -> Path | None:
    runs = []
    for p in save_dir.iterdir():
        if not p.is_dir():
            continue
        summary = p / "summary.json"
        if not summary.exists():
            continue
        if p.stat().st_mtime >= after_ts:
            runs.append(p)
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def summary_brief(summary_path: Path) -> Dict[str, float]:
    j = json.loads(summary_path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    if "gesture" in j and "summary" in j["gesture"]:
        out["gesture_accuracy_pct"] = j["gesture"]["summary"].get("accuracy_pct", 0.0)
    if "voice" in j and "summary" in j["voice"]:
        out["voice_accuracy_pct"] = j["voice"]["summary"].get("accuracy_pct", 0.0)
    return out


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pl_sweep: List[float] = []
    for tok in args.pl_clock_sweep.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pl_sweep.append(float(tok))
        except ValueError:
            print(f"[WARN] Ignoring invalid --pl-clock-sweep token: {tok!r}")

    plan: List[Tuple[str, List[str]]] = [
        (
            f"{args.tag_prefix}_gesture",
            ["--mode", "gesture", "--gesture-pack", "q88"],
        ),
        (
            f"{args.tag_prefix}_voice",
            [
                "--mode",
                "voice",
                "--voice-pack",
                "q88",
                "--voice-order",
                args.voice_order,
            ],
        ),
        (
            f"{args.tag_prefix}_both",
            [
                "--mode",
                "both",
                "--gesture-pack",
                "q88",
                "--voice-pack",
                "q88",
                "--voice-order",
                args.voice_order,
            ],
        ),
        (
            f"{args.tag_prefix}_cpu_perf",
            [
                "--mode",
                "both",
                "--gesture-pack",
                "q88",
                "--voice-pack",
                "q88",
                "--voice-order",
                args.voice_order,
                "--cpu-governor",
                "performance",
            ],
        ),
        (
            f"{args.tag_prefix}_cpu_freq",
            [
                "--mode",
                "both",
                "--gesture-pack",
                "q88",
                "--voice-pack",
                "q88",
                "--voice-order",
                args.voice_order,
                "--cpu-governor",
                "userspace",
                "--cpu-freq-khz",
                str(args.cpu_freq_khz),
            ],
        ),
    ]

    for mhz in pl_sweep:
        tag_mhz = str(int(mhz)) if float(mhz).is_integer() else str(mhz).replace(".", "p")
        plan.append(
            (
                f"{args.tag_prefix}_pl{tag_mhz}",
                [
                    "--mode",
                    "both",
                    "--gesture-pack",
                    "q88",
                    "--voice-pack",
                    "q88",
                    "--voice-order",
                    args.voice_order,
                    "--pl-clock-mhz",
                    str(mhz),
                ],
            )
        )

    if not args.skip_power:
        plan.append(
            (
                f"{args.tag_prefix}_power_manual",
                [
                    "--mode",
                    "both",
                    "--gesture-pack",
                    "q88",
                    "--voice-pack",
                    "q88",
                    "--voice-order",
                    args.voice_order,
                    "--cpu-governor",
                    args.cpu_governor,
                    "--pl-clock-mhz",
                    str(args.pl_clock_mhz),
                    "--power-w",
                    str(args.power_w),
                ],
            )
        )
        plan.append(
            (
                f"{args.tag_prefix}_power_sysfs",
                [
                    "--mode",
                    "both",
                    "--gesture-pack",
                    "q88",
                    "--voice-pack",
                    "q88",
                    "--voice-order",
                    args.voice_order,
                    "--cpu-governor",
                    args.cpu_governor,
                    "--pl-clock-mhz",
                    str(args.pl_clock_mhz),
                    "--power-sysfs-path",
                    args.power_sysfs_path,
                    "--power-sysfs-scale",
                    str(args.power_sysfs_scale),
                ],
            )
        )

    results = []
    failed = False

    for tag, extra in plan:
        start_ts = time.time()
        rc = run_one(
            python_bin=args.python_bin,
            runner=args.runner,
            save_dir=str(save_dir),
            xsa_path=args.xsa_path,
            tag=tag,
            extra=extra,
        )
        run_dir = newest_summary_dir(save_dir, after_ts=start_ts)
        summary_path = (run_dir / "summary.json") if run_dir else None
        row = {
            "tag": tag,
            "return_code": rc,
            "run_dir": str(run_dir) if run_dir else "",
            "summary_path": str(summary_path) if summary_path else "",
        }
        if summary_path and summary_path.exists():
            row["metrics"] = summary_brief(summary_path)
        results.append(row)

        if rc != 0:
            failed = True
            print(f"[FAIL] tag={tag} rc={rc}")
        else:
            print(f"[OK]   tag={tag}")

    bundle = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "python_bin": args.python_bin,
            "runner": args.runner,
            "xsa_path": args.xsa_path,
            "save_dir": str(save_dir),
            "voice_order": args.voice_order,
            "skip_power": args.skip_power,
            "cpu_governor": args.cpu_governor,
            "cpu_freq_khz": args.cpu_freq_khz,
            "pl_clock_mhz": args.pl_clock_mhz,
            "pl_clock_sweep": args.pl_clock_sweep,
            "power_w": args.power_w,
            "power_sysfs_path": args.power_sysfs_path,
            "power_sysfs_scale": args.power_sysfs_scale,
        },
        "runs": results,
    }

    bundle_path = save_dir / f"assessment_bundle_{time.strftime('%Y%m%d_%H%M%S')}.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved bundle: {bundle_path}")

    if failed:
        print("At least one run failed. Check the run logs above and summary paths in bundle.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
