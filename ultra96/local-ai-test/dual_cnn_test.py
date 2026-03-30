from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pynq import Overlay, allocate


# -------------------------
# Project defaults/constants
# -------------------------
DEFAULT_XSA = "dual_cnn.xsa"
DEFAULT_GESTURE_X = "gesture_X_test.npy"
DEFAULT_GESTURE_Y = "gesture_y_test.npy"
DEFAULT_VOICE_X = "voice_X_test.npy"
DEFAULT_VOICE_Y = "voice_y_test.npy"
DEFAULT_OUTDIR = "report/evidence_dual"

GESTURE_CORE_NAME = "gesture_cnn_0"
VOICE_CORE_NAME = "voice_cnn_0"
GESTURE_DMA_NAME = "axi_dma_1"
VOICE_DMA_NAME = "axi_dma_0"

GESTURE_LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
VOICE_LABELS = ["Bulbasaur", "Charizard", "Pikachu"]

# Current class meanings used by the latest datasets/model wiring:
# Gesture: 0=Raise, 1=Shake, 2=Chop, 3=Stir, 4=Swing, 5=Punch
# Voice: 0=bulbasaur, 1=charizard, 2=pikachu


@dataclass
class EvalResult:
    total: int
    correct: int
    cm: np.ndarray
    total_ms: List[float]
    prep_ms: List[float]
    ctrl_ms: List[float]
    comm_ms: List[float]
    inference_ms: List[float]
    dma_submit_ms: List[float]
    dma_wait_ms: List[float]

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


# -------------------------
# Hardware helper functions
# -------------------------
def q88_pack_u32(x: np.ndarray) -> np.ndarray:
    """Pack float values to signed Q8.8 in AXIS data[15:0] (uint32 words)."""
    q = np.round(np.clip(x, -128.0, 127.99609375) * 256.0).astype(np.int32)
    return (q & 0xFFFF).astype(np.uint32)


def reset_dma(dma) -> None:
    # Reset MM2S and S2MM channels, then return both to run state.
    for ch in [dma.sendchannel, dma.recvchannel]:
        ch._mmio.write(ch._offset + 0x00, 0x4)
    time.sleep(0.01)
    for ch in [dma.sendchannel, dma.recvchannel]:
        ch._mmio.write(ch._offset + 0x00, 0x1)
    time.sleep(0.01)


def run_dma(dma, in_buf, out_buf, timeout_s: float) -> Tuple[float, float, float]:
    in_buf.flush()
    out_buf.invalidate()

    # Start RX first to avoid initial backpressure.
    t_submit0 = time.perf_counter()
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)
    submit_ms = (time.perf_counter() - t_submit0) * 1000.0

    t_wait0 = time.perf_counter()
    while True:
        if dma.sendchannel.idle and dma.recvchannel.idle:
            break
        if time.perf_counter() - t_wait0 > timeout_s:
            raise TimeoutError("DMA timeout")
        # Avoid fixed sleep granularity here; it can dominate short-latency measurements.
        pass

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    wait_ms = (time.perf_counter() - t_wait0) * 1000.0
    out_buf.invalidate()
    return submit_ms, wait_ms, submit_ms + wait_ms


def stop_core(core) -> None:
    core.write(0x00, 0x00)


def start_core(core) -> None:
    core.write(0x00, 0x01)


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def read_total_power_w() -> Tuple[Optional[float], List[str]]:
    """Read and sum all hwmon rail power*_input files, returning Watts."""
    paths = sorted(Path("/sys/class/hwmon").glob("hwmon*/power*_input"))
    total_uw = 0.0
    used_paths: List[str] = []
    for p in paths:
        txt = _read_text(str(p))
        if txt is None:
            continue
        try:
            total_uw += float(txt)
            used_paths.append(str(p))
        except ValueError:
            continue
    if not used_paths:
        return None, []
    return total_uw * 1e-6, used_paths


def get_cpu_governor() -> Optional[str]:
    return _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def get_cpu_freq_khz() -> Optional[int]:
    txt = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if txt is None:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def set_cpu_governor(governor: str) -> bool:
    ok = True
    gov_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    if not gov_paths:
        return False
    for path in gov_paths:
        try:
            path.write_text(governor, encoding="utf-8")
        except Exception:
            ok = False
    return ok


def set_cpu_freq_khz(freq_khz: int) -> bool:
    # Preferred path: userspace governor + scaling_setspeed.
    set_paths = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_setspeed"))
    if not set_paths:
        set_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_setspeed"))
    if set_paths:
        ok = True
        for path in set_paths:
            try:
                path.write_text(str(freq_khz), encoding="utf-8")
            except Exception:
                ok = False
        if ok:
            return True

    # Fallback path: pin min=max for current policy (works on many cpufreq-dt setups).
    min_paths = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_min_freq"))
    max_paths = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_max_freq"))
    if not min_paths or not max_paths or len(min_paths) != len(max_paths):
        return False

    ok = True
    for min_p, max_p in zip(min_paths, max_paths):
        try:
            min_p.write_text(str(freq_khz), encoding="utf-8")
            max_p.write_text(str(freq_khz), encoding="utf-8")
        except Exception:
            ok = False
    return ok


def set_pl_clock_mhz(clock_mhz: float) -> bool:
    try:
        from pynq.ps import Clocks  # type: ignore

        Clocks.fclk0_mhz = float(clock_mhz)
        return True
    except Exception:
        return False


def get_pl_clock_mhz() -> Optional[float]:
    try:
        from pynq.ps import Clocks  # type: ignore

        return float(Clocks.fclk0_mhz)
    except Exception:
        return None


# -------------------------
# Data loading
# -------------------------
def read_gesture_npy(
    features_path: str,
    labels_path: str,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load gesture windows and normalize layout to [N, 60, 6]."""
    X = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"Gesture npy must be rank-3, got {X.shape}")
    if X.shape[1:] == (60, 6):
        Xw = X
    elif X.shape[1:] == (6, 60):
        Xw = np.transpose(X, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Gesture npy shape must be [N,60,6] or [N,6,60], got {X.shape}")

    n = min(len(Xw), len(y))
    if max_samples > 0:
        n = min(n, max_samples)
    Xw = Xw[:n]
    y = y[:n]

    # Keep only labels represented by current 6-class gesture model.
    valid = (y >= 0) & (y < len(GESTURE_LABELS))
    Xw = Xw[valid]
    y = y[valid]
    if len(Xw) == 0:
        raise RuntimeError("No valid gesture samples after label filtering.")
    return Xw, y


def read_voice_npy(
    features_path: str,
    labels_path: str,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64).reshape(-1)
    n = min(len(X), len(y))
    if max_samples > 0:
        n = min(n, max_samples)
    return X[:n], y[:n]


# -------------------------
# Evaluation loops
# -------------------------
def run_gesture_eval(gesture_core, gesture_dma, X: np.ndarray, y: np.ndarray, timeout_s: float) -> EvalResult:
    n = len(X)
    num_classes = len(GESTURE_LABELS)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    total_ms: List[float] = []
    prep_ms: List[float] = []
    ctrl_ms: List[float] = []
    comm_ms: List[float] = []
    inference_ms: List[float] = []
    dma_submit_ms: List[float] = []
    dma_wait_ms: List[float] = []
    correct = 0

    in_buffer = allocate(shape=(60 * 6,), dtype=np.uint32)
    out_buffer = allocate(shape=(1,), dtype=np.uint32)

    try:
        for i in range(n):
            t_total0 = time.perf_counter()

            # Gesture stream order is [time][channel], flattened.
            t_prep0 = time.perf_counter()
            flat = X[i].reshape(-1).astype(np.float32)
            np.copyto(in_buffer, q88_pack_u32(flat))
            prep = (time.perf_counter() - t_prep0) * 1000.0

            t_ctrl0 = time.perf_counter()
            start_core(gesture_core)
            ctrl = (time.perf_counter() - t_ctrl0) * 1000.0
            dma_submit, dma_wait, comm = run_dma(gesture_dma, in_buffer, out_buffer, timeout_s=timeout_s)
            total = (time.perf_counter() - t_total0) * 1000.0

            prep_ms.append(prep)
            ctrl_ms.append(ctrl)
            comm_ms.append(comm)
            inference_ms.append(ctrl + comm)
            total_ms.append(total)
            dma_submit_ms.append(dma_submit)
            dma_wait_ms.append(dma_wait)

            pred = int(out_buffer[0])
            if pred < 0 or pred >= num_classes:
                pred = 0

            yt = int(y[i])
            cm[yt, pred] += 1
            correct += int(pred == yt)
    finally:
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass

    return EvalResult(
        total=n,
        correct=correct,
        cm=cm,
        total_ms=total_ms,
        prep_ms=prep_ms,
        ctrl_ms=ctrl_ms,
        comm_ms=comm_ms,
        inference_ms=inference_ms,
        dma_submit_ms=dma_submit_ms,
        dma_wait_ms=dma_wait_ms,
    )


def run_voice_eval(voice_core, voice_dma, X: np.ndarray, y: np.ndarray, timeout_s: float) -> EvalResult:
    n = len(X)
    num_classes = len(VOICE_LABELS)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    total_ms: List[float] = []
    prep_ms: List[float] = []
    ctrl_ms: List[float] = []
    comm_ms: List[float] = []
    inference_ms: List[float] = []
    dma_submit_ms: List[float] = []
    dma_wait_ms: List[float] = []
    correct = 0

    in_buffer = allocate(shape=(40 * 50,), dtype=np.uint32)
    out_buffer = allocate(shape=(1,), dtype=np.uint32)

    try:
        for i in range(n):
            t_total0 = time.perf_counter()

            # Voice model expects [time][channel], i.e. transpose [40,50] -> [50,40].
            t_prep0 = time.perf_counter()
            feat = X[i].astype(np.float32)
            flat = feat.T.reshape(-1).astype(np.float32)
            np.copyto(in_buffer, q88_pack_u32(flat))
            prep = (time.perf_counter() - t_prep0) * 1000.0

            t_ctrl0 = time.perf_counter()
            start_core(voice_core)
            ctrl = (time.perf_counter() - t_ctrl0) * 1000.0
            dma_submit, dma_wait, comm = run_dma(voice_dma, in_buffer, out_buffer, timeout_s=timeout_s)
            total = (time.perf_counter() - t_total0) * 1000.0

            prep_ms.append(prep)
            ctrl_ms.append(ctrl)
            comm_ms.append(comm)
            inference_ms.append(ctrl + comm)
            total_ms.append(total)
            dma_submit_ms.append(dma_submit)
            dma_wait_ms.append(dma_wait)

            pred = int(out_buffer[0])
            if pred < 0 or pred >= num_classes:
                pred = 0
            yt = int(y[i])
            if yt < 0 or yt >= num_classes:
                continue
            cm[yt, pred] += 1
            correct += int(pred == yt)
    finally:
        try:
            in_buffer.close()
            out_buffer.close()
        except Exception:
            pass

    return EvalResult(
        total=n,
        correct=correct,
        cm=cm,
        total_ms=total_ms,
        prep_ms=prep_ms,
        ctrl_ms=ctrl_ms,
        comm_ms=comm_ms,
        inference_ms=inference_ms,
        dma_submit_ms=dma_submit_ms,
        dma_wait_ms=dma_wait_ms,
    )


# -------------------------
# Reporting helpers
# -------------------------
def _stats(values: Sequence[float], key: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{key}_mean_ms": 0.0,
            f"{key}_p50_ms": 0.0,
            f"{key}_p90_ms": 0.0,
            f"{key}_p99_ms": 0.0,
            f"{key}_max_ms": 0.0,
        }
    return {
        f"{key}_mean_ms": float(arr.mean()),
        f"{key}_p50_ms": float(np.percentile(arr, 50)),
        f"{key}_p90_ms": float(np.percentile(arr, 90)),
        f"{key}_p99_ms": float(np.percentile(arr, 99)),
        f"{key}_max_ms": float(arr.max()),
    }


def summarize(result: EvalResult, power_w: Optional[float] = None) -> Dict[str, float]:
    out: Dict[str, float] = {
        "total": int(result.total),
        "correct": int(result.correct),
        "accuracy_pct": float(result.accuracy),
    }
    out.update(_stats(result.total_ms, "latency_total"))
    out.update(_stats(result.prep_ms, "latency_prep"))
    out.update(_stats(result.ctrl_ms, "latency_ctrl"))
    out.update(_stats(result.comm_ms, "latency_comm"))
    out.update(_stats(result.inference_ms, "latency_inference"))
    out.update(_stats(result.dma_submit_ms, "latency_dma_submit"))
    out.update(_stats(result.dma_wait_ms, "latency_dma_wait"))

    # Backward-compatible aliases used by older scripts/reports.
    out["latency_mean_ms"] = out["latency_total_mean_ms"]
    out["latency_p50_ms"] = out["latency_total_p50_ms"]
    out["latency_p90_ms"] = out["latency_total_p90_ms"]
    out["latency_p99_ms"] = out["latency_total_p99_ms"]
    out["latency_max_ms"] = out["latency_total_max_ms"]

    if power_w is not None:
        out["power_w"] = float(power_w)
        # 1 W * 1 ms = 1 mJ.
        out["energy_total_mean_mj"] = float(power_w) * out["latency_total_mean_ms"]
        out["energy_inference_mean_mj"] = float(power_w) * out["latency_inference_mean_ms"]
        out["energy_comm_mean_mj"] = float(power_w) * out["latency_comm_mean_ms"]
    return out


def pretty_cm(cm: np.ndarray, labels: Sequence[str]) -> None:
    header = "true\\pred | " + " ".join([f"{l[:6]:>6}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = " ".join([f"{int(cm[i, j]):6d}" for j in range(len(labels))])
        print(f"{label[:9]:>9} | {row}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual CNN hardware test on Ultra96 (PYNQ)")
    p.add_argument("--xsa-path", default=DEFAULT_XSA)
    p.add_argument("--mode", choices=["gesture", "voice", "both"], default="both")

    p.add_argument("--gesture-features", default=DEFAULT_GESTURE_X)
    p.add_argument("--gesture-labels", default=DEFAULT_GESTURE_Y)
    p.add_argument("--voice-features", default=DEFAULT_VOICE_X)
    p.add_argument("--voice-labels", default=DEFAULT_VOICE_Y)
    p.add_argument("--gesture-max-samples", type=int, default=120)
    p.add_argument("--voice-max-samples", type=int, default=300)

    p.add_argument("--timeout-s", type=float, default=2.0)
    p.add_argument("--save-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--tag", default="")
    p.add_argument("--cpu-governor", default=None, help="Set Linux CPU governor, e.g. userspace/performance.")
    p.add_argument("--cpu-freq-khz", type=int, default=None, help="Set PS CPU frequency in kHz.")
    p.add_argument("--pl-clock-mhz", type=float, default=None, help="Set PL FCLK0 frequency in MHz.")
    p.add_argument(
        "--power",
        action="store_true",
        help="Sum all hwmon power rails and report total power/energy in Watts/mJ.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id = f"{run_id}_{args.tag}"
    out_dir = Path(args.save_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rail_paths: List[str] = []
    power_w: Optional[float] = None
    if args.power:
        power_w, rail_paths = read_total_power_w()
        if power_w is None:
            print("[WARN] --power enabled but no readable hwmon power*_input rails were found.")
    if power_w is not None:
        print(f"Using power for energy estimates: {power_w:.4f} W")

    runtime_controls = {
        "requested": {
            "cpu_governor": args.cpu_governor,
            "cpu_freq_khz": args.cpu_freq_khz,
            "pl_clock_mhz": args.pl_clock_mhz,
        },
        "before": {
            "cpu_governor": get_cpu_governor(),
            "cpu_freq_khz": get_cpu_freq_khz(),
            "pl_clock_mhz": get_pl_clock_mhz(),
        },
        "applied": {
            "cpu_governor_set": False,
            "cpu_freq_set": False,
            "pl_clock_set": False,
        },
        "after": {},
        "warnings": [],
    }

    if args.cpu_governor:
        ok = set_cpu_governor(args.cpu_governor)
        runtime_controls["applied"]["cpu_governor_set"] = ok
        if not ok:
            runtime_controls["warnings"].append("Unable to set CPU governor.")

    if args.cpu_freq_khz is not None:
        ok = set_cpu_freq_khz(args.cpu_freq_khz)
        runtime_controls["applied"]["cpu_freq_set"] = ok
        if not ok:
            runtime_controls["warnings"].append(
                "Unable to set CPU frequency (setspeed/min-max paths unavailable or permission denied)."
            )

    print(f"Loading overlay: {args.xsa_path}")
    overlay = Overlay(args.xsa_path)
    print("Available IP blocks:", ", ".join(sorted(overlay.ip_dict.keys())))

    # Apply PL clock after overlay load so bitstream load does not overwrite the request.
    if args.pl_clock_mhz is not None:
        ok = set_pl_clock_mhz(args.pl_clock_mhz)
        runtime_controls["applied"]["pl_clock_set"] = ok
        if not ok:
            runtime_controls["warnings"].append("Unable to set PL clock.")

    runtime_controls["after"] = {
        "cpu_governor": get_cpu_governor(),
        "cpu_freq_khz": get_cpu_freq_khz(),
        "pl_clock_mhz": get_pl_clock_mhz(),
    }
    after_cpu = runtime_controls["after"]["cpu_freq_khz"]
    if after_cpu is not None:
        print(f"CPU freq readback: {int(after_cpu)} kHz")
    after_pl = runtime_controls["after"]["pl_clock_mhz"]
    if after_pl is not None:
        print(f"PL clock readback (FCLK0): {float(after_pl):.2f} MHz")

    for msg in runtime_controls["warnings"]:
        print("[WARN]", msg)

    gesture_core = getattr(overlay, GESTURE_CORE_NAME)
    voice_core = getattr(overlay, VOICE_CORE_NAME)
    gesture_dma = getattr(overlay, GESTURE_DMA_NAME)
    voice_dma = getattr(overlay, VOICE_DMA_NAME)

    # Start from a known clean state.
    stop_core(gesture_core)
    stop_core(voice_core)
    reset_dma(gesture_dma)
    if voice_dma is not gesture_dma:
        reset_dma(voice_dma)

    report: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "runtime_controls": runtime_controls,
        "overlay": {
            "gesture_core": GESTURE_CORE_NAME,
            "voice_core": VOICE_CORE_NAME,
            "gesture_dma": GESTURE_DMA_NAME,
            "voice_dma": VOICE_DMA_NAME,
            "input_pack": "q88",
            "voice_input_order": "tc",
        },
        "power": {
            "enabled": args.power,
            "sum_rails_w": power_w,
            "rail_paths": rail_paths,
        },
    }

    try:
        if args.mode in ("gesture", "both"):
            Xg, yg = read_gesture_npy(
                features_path=args.gesture_features,
                labels_path=args.gesture_labels,
                max_samples=args.gesture_max_samples,
            )
            print(f"\nRunning gesture test: {len(Xg)} samples")
            gres = run_gesture_eval(gesture_core, gesture_dma, Xg, yg, timeout_s=args.timeout_s)
            gsum = summarize(gres, power_w=power_w)

            print("Gesture accuracy:", f"{gsum['accuracy_pct']:.2f}%")
            print(
                "Gesture timing mean(ms):",
                f"total={gsum['latency_total_mean_ms']:.3f},",
                f"inference={gsum['latency_inference_mean_ms']:.3f},",
                f"comm={gsum['latency_comm_mean_ms']:.3f},",
                f"prep={gsum['latency_prep_mean_ms']:.3f}",
            )
            pretty_cm(gres.cm, GESTURE_LABELS)
            report["gesture"] = {
                "summary": gsum,
                "labels": GESTURE_LABELS,
                "confusion_matrix": gres.cm.tolist(),
            }
            stop_core(gesture_core)

        if args.mode in ("voice", "both"):
            Xv, yv = read_voice_npy(
                features_path=args.voice_features,
                labels_path=args.voice_labels,
                max_samples=args.voice_max_samples,
            )
            print(f"\nRunning voice test: {len(Xv)} samples")
            vres = run_voice_eval(voice_core, voice_dma, Xv, yv, timeout_s=args.timeout_s)
            vsum = summarize(vres, power_w=power_w)

            print("Voice accuracy:", f"{vsum['accuracy_pct']:.2f}%")
            print(
                "Voice timing mean(ms):",
                f"total={vsum['latency_total_mean_ms']:.3f},",
                f"inference={vsum['latency_inference_mean_ms']:.3f},",
                f"comm={vsum['latency_comm_mean_ms']:.3f},",
                f"prep={vsum['latency_prep_mean_ms']:.3f}",
            )
            pretty_cm(vres.cm, VOICE_LABELS)
            report["voice"] = {
                "summary": vsum,
                "confusion_matrix": vres.cm.tolist(),
            }
            stop_core(voice_core)
    finally:
        # Always stop both cores even if an exception occurs.
        stop_core(gesture_core)
        stop_core(voice_core)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {summary_path}")


if __name__ == "__main__":
    main()
