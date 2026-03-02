from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run assessment commands using dual_cnn_test.py and collect summaries."
    )
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")
    p.add_argument("--runner", default="dual_cnn_test.py", help="Path to dual_cnn_test.py.")
    p.add_argument("--xsa-path", default="dual_cnn.xsa")
    p.add_argument("--save-dir", default="../report/evidence_dual")
    p.add_argument("--tag-prefix", default="assess", help="Prefix used for run tags.")
    p.add_argument("--gesture-max-samples", type=int, default=120)
    p.add_argument("--voice-max-samples", type=int, default=300)
    p.add_argument("--timeout-s", type=float, default=2.0)
    p.add_argument("--pl-clock-sweep", default="75,100,125", help="Comma-separated PL clock sweep values in MHz.")
    p.add_argument(
        "--ps-freq-sweep-khz",
        default="1000000,1200000,1400000",
        help="Comma-separated PS CPU frequency sweep values in kHz.",
    )
    p.add_argument("--power-w", type=float, default=4.2)
    p.add_argument("--power-sysfs-path", default="/sys/class/hwmon/hwmon0/power1_input")
    p.add_argument("--power-sysfs-scale", type=float, default=1e-6)
    p.add_argument("--skip-power", action="store_true", help="Skip power-profiled runs.")
    return p.parse_args()


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
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    if "gesture" in data and "summary" in data["gesture"]:
        out["gesture_accuracy_pct"] = data["gesture"]["summary"].get("accuracy_pct", 0.0)
    if "voice" in data and "summary" in data["voice"]:
        out["voice_accuracy_pct"] = data["voice"]["summary"].get("accuracy_pct", 0.0)
    return out


def build_commands(args: argparse.Namespace, save_dir: Path) -> List[Dict[str, object]]:
    # Shared arguments across all runs.
    common = [
        args.python_bin,
        args.runner,
        "--xsa-path",
        args.xsa_path,
        "--save-dir",
        str(save_dir),
        "--gesture-max-samples",
        str(args.gesture_max_samples),
        "--voice-max-samples",
        str(args.voice_max_samples),
        "--timeout-s",
        str(args.timeout_s),
    ]

    pl_sweep: List[float] = []
    for tok in args.pl_clock_sweep.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pl_sweep.append(float(tok))
        except ValueError:
            print(f"[WARN] Ignoring invalid --pl-clock-sweep token: {tok!r}")

    ps_sweep: List[int] = []
    for tok in args.ps_freq_sweep_khz.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ps_sweep.append(int(tok))
        except ValueError:
            print(f"[WARN] Ignoring invalid --ps-freq-sweep-khz token: {tok!r}")

    # Plain-text commands equivalent to what this script runs (with defaults):
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_gesture --mode gesture
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_voice --mode voice
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_both --mode both
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_pl100 --mode both --pl-clock-mhz 100
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_ps1200000 --mode both --cpu-governor userspace --cpu-freq-khz 1200000
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_power_manual --mode both --power-w 4.2
    # python dual_cnn_test.py --xsa-path dual_cnn.xsa --save-dir ../report/evidence_dual --gesture-max-samples 120 --voice-max-samples 300 --timeout-s 2.0 --tag assess_power_sysfs --mode both --power-sysfs-path /sys/class/hwmon/hwmon0/power1_input --power-sysfs-scale 1e-6
    #
    # Yes, you can run these plain-text commands directly for assessment.
    # This script is just a wrapper that runs them and bundles results.

    # Readable list of concrete CLI commands that will be executed.
    commands: List[Dict[str, object]] = [
        {
            "tag": f"{args.tag_prefix}_gesture",
            "mode": "gesture",
            "cmd": common + ["--tag", f"{args.tag_prefix}_gesture", "--mode", "gesture"],
        },
        {
            "tag": f"{args.tag_prefix}_voice",
            "mode": "voice",
            "cmd": common + ["--tag", f"{args.tag_prefix}_voice", "--mode", "voice"],
        },
        {
            "tag": f"{args.tag_prefix}_both",
            "mode": "both",
            "cmd": common + ["--tag", f"{args.tag_prefix}_both", "--mode", "both"],
        },
    ]

    for mhz in pl_sweep:
        tag_mhz = str(int(mhz)) if float(mhz).is_integer() else str(mhz).replace(".", "p")
        commands.append(
            {
                "tag": f"{args.tag_prefix}_pl{tag_mhz}",
                "mode": "both",
                "cmd": common
                + [
                    "--tag",
                    f"{args.tag_prefix}_pl{tag_mhz}",
                    "--mode",
                    "both",
                    "--pl-clock-mhz",
                    str(mhz),
                ],
            }
        )

    for khz in ps_sweep:
        commands.append(
            {
                "tag": f"{args.tag_prefix}_ps{khz}",
                "mode": "both",
                "cmd": common
                + [
                    "--tag",
                    f"{args.tag_prefix}_ps{khz}",
                    "--mode",
                    "both",
                    "--cpu-governor",
                    "userspace",
                    "--cpu-freq-khz",
                    str(khz),
                ],
            }
        )

    if not args.skip_power:
        commands.extend(
            [
                {
                    "tag": f"{args.tag_prefix}_power_manual",
                    "mode": "both",
                    "cmd": common
                    + [
                        "--tag",
                        f"{args.tag_prefix}_power_manual",
                        "--mode",
                        "both",
                        "--power-w",
                        str(args.power_w),
                    ],
                },
                {
                    "tag": f"{args.tag_prefix}_power_sysfs",
                    "mode": "both",
                    "cmd": common
                    + [
                        "--tag",
                        f"{args.tag_prefix}_power_sysfs",
                        "--mode",
                        "both",
                        "--power-sysfs-path",
                        args.power_sysfs_path,
                        "--power-sysfs-scale",
                        str(args.power_sysfs_scale),
                    ],
                },
            ]
        )

    return commands


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    commands = build_commands(args, save_dir)
    results = []
    failed = False

    for entry in commands:
        cmd = entry["cmd"]
        assert isinstance(cmd, list)
        tag = str(entry["tag"])
        mode = str(entry["mode"])

        start_ts = time.time()
        print("\n[RUN]", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode

        run_dir = newest_summary_dir(save_dir, after_ts=start_ts)
        summary_path = (run_dir / "summary.json") if run_dir else None

        row = {
            "tag": tag,
            "mode": mode,
            "return_code": rc,
            "run_dir": str(run_dir) if run_dir else "",
            "summary_path": str(summary_path) if summary_path else "",
            "command": cmd,
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
            "gesture_max_samples": args.gesture_max_samples,
            "voice_max_samples": args.voice_max_samples,
            "timeout_s": args.timeout_s,
            "pl_clock_sweep": args.pl_clock_sweep,
            "ps_freq_sweep_khz": args.ps_freq_sweep_khz,
            "power_w": args.power_w,
            "power_sysfs_path": args.power_sysfs_path,
            "power_sysfs_scale": args.power_sysfs_scale,
            "skip_power": args.skip_power,
        },
        "runs": results,
    }

    bundle_path = save_dir / f"assessment_bundle_{time.strftime('%Y%m%d_%H%M%S')}.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved bundle: {bundle_path}")

    if failed:
        print("At least one run failed. Check run logs and summary paths in the bundle.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
