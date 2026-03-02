# preprocess.py
import numpy as np
from collections import deque

class GesturePreprocessor:
    """
    Turns streaming IMU samples at fs Hz into a fixed 60x6 window:
      1) (optional) segment using motion detection or button boundaries
      2) resample to 60 timesteps (linear interpolation)
      3) z-score using training mean/std (per-feature)
      4) flatten to 360 floats, time-major: [t0(6), t1(6), ...]
    """

    def __init__(
        self,
        fs=50,
        target_len=60,
        max_len_s=2.0,
        min_len_s=0.25,
        # motion detection parameters (can be tuned later)
        ema_alpha=0.05,          # for gravity/acc baseline
        beta_acc=0.5,            # weight for acc-change part in energy
        start_k=3,               # consecutive samples above start threshold
        end_m=8,                 # consecutive samples below end threshold
        idle_calib_s=1.0,        # seconds used to calibrate thresholds
        thr_start_z=3.0,         # start threshold = mu + thr_start_z*sd
        thr_end_z=1.5,           # end threshold   = mu + thr_end_z*sd
    ):
        self.fs = fs
        self.target_len = target_len
        self.max_len = int(round(max_len_s * fs))         # e.g. 100
        self.min_len = max(2, int(round(min_len_s * fs))) # e.g. 13

        # scaler (training stats)
        self.mean = np.array([-0.79198196, -0.89734741, -1.85433716, -0.04240223, -0.00923926, -0.00578469], dtype=np.float32)  
        self.std = np.array([ 55.806826,   103.93487478,  99.69469418,   1.1262504,    1.12243429, 1.03791274], dtype=np.float32)

        # streaming buffers
        self.ring = deque(maxlen=self.max_len * 3)  # keep more than max_len for continuous scanning
        self.segment = []                           # current segment samples (list of 6D)
        self.recording = False                      # button capture flag

        # acc baseline (for "delta acc" energy)
        self.ema_alpha = ema_alpha
        self.acc_ema = None

        # motion thresholds (auto-calibrated from idle)
        self.beta_acc = beta_acc
        self.idle_calib_n = int(round(idle_calib_s * fs))
        self.idle_energy = []
        self.energy_mu = None
        self.energy_sd = None
        self.thr_start_z = thr_start_z
        self.thr_end_z = thr_end_z
        self.thr_start = None
        self.thr_end = None

        # hysteresis counters
        self.start_k = start_k
        self.end_m = end_m
        self.above_cnt = 0
        self.below_cnt = 0

        # state for continuous mode
        self.in_motion = False

    # -----------------------------
    # Public API
    # -----------------------------
    def load_scaler(self, mean_path="mean.npy", std_path="std.npy"):
        mean = np.load(mean_path).astype(np.float32)
        std  = np.load(std_path).astype(np.float32)

        # accept (6,) only (recommended for IMU)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError(f"Expected mean/std shape (6,), got {mean.shape} and {std.shape}")

        self.mean = mean
        self.std = std
        return self.mean, self.std

    # Button mode: call start_recording(), then feed samples via push(),
    # then stop_recording() to get final processed window.
    def start_recording(self):
        self.recording = True
        self.segment = []

    def stop_recording(self):
        self.recording = False
        if len(self.segment) < self.min_len:
            return None  # too short
        return self._finalize_segment(np.array(self.segment, dtype=np.float32))

    # Continuous mode: call push(sample) repeatedly; it returns a window when a gesture is detected & ended.
    def push(self, sample6):
        """
        sample6: array-like length 6 in THIS ORDER:
          [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        Returns:
          None, or dict with:
            {
              "flat360": np.float32[360],
              "window60x6": np.float32[60,6],
              "rawNx6": np.float32[N,6],
              "N": int
            }
        """
        x = np.asarray(sample6, dtype=np.float32).reshape(6,)
        self.ring.append(x)

        # Update acc EMA baseline for delta-acc energy
        acc = x[3:6]
        if self.acc_ema is None:
            self.acc_ema = acc.copy()
        else:
            self.acc_ema = (1.0 - self.ema_alpha) * self.acc_ema + self.ema_alpha * acc

        # If button recording is on, just accumulate and do no auto-detection
        if self.recording:
            self.segment.append(x)
            # Hard cap to avoid runaway
            if len(self.segment) > self.max_len:
                # auto-stop at max_len
                self.recording = False
                return self._finalize_segment(np.array(self.segment, dtype=np.float32))
            return None

        # Continuous: compute motion energy
        e = self._energy(x)

        # Auto-calibrate thresholds from initial idle period
        if self.thr_start is None or self.thr_end is None:
            self.idle_energy.append(e)
            if len(self.idle_energy) >= self.idle_calib_n:
                arr = np.array(self.idle_energy, dtype=np.float32)
                self.energy_mu = float(arr.mean())
                self.energy_sd = float(arr.std() + 1e-6)
                self.thr_start = self.energy_mu + self.thr_start_z * self.energy_sd
                self.thr_end   = self.energy_mu + self.thr_end_z   * self.energy_sd
            return None

        # Hysteresis logic
        if not self.in_motion:
            # waiting for gesture start
            if e > self.thr_start:
                self.above_cnt += 1
            else:
                self.above_cnt = 0

            if self.above_cnt >= self.start_k:
                self.in_motion = True
                self.segment = []
                self.below_cnt = 0
                self.above_cnt = 0
                # include a small pre-roll from ring to avoid clipping start
                pre = min(len(self.ring), int(0.15 * self.fs))  # 150ms pre-roll
                if pre > 0:
                    for s in list(self.ring)[-pre:]:
                        self.segment.append(s)
            return None

        else:
            # capturing gesture
            self.segment.append(x)

            # stop condition
            if e < self.thr_end:
                self.below_cnt += 1
            else:
                self.below_cnt = 0

            # safety cap
            if len(self.segment) >= self.max_len:
                self.in_motion = False
                self.below_cnt = 0
                return self._finalize_segment(np.array(self.segment, dtype=np.float32))

            if self.below_cnt >= self.end_m:
                self.in_motion = False
                self.below_cnt = 0

                raw = np.array(self.segment, dtype=np.float32)
                if raw.shape[0] < self.min_len:
                    return None  # ignore too-short triggers
                return self._finalize_segment(raw)

            return None

    # -----------------------------
    # Internals
    # -----------------------------
    def _energy(self, x):
        # gyro magnitude
        g = x[0:3]
        gyro_mag = float(np.sqrt(np.sum(g * g)))

        # delta acc magnitude (remove slow baseline)
        acc = x[3:6]
        dacc = acc - self.acc_ema
        dacc_mag = float(np.sqrt(np.sum(dacc * dacc)))

        return gyro_mag + self.beta_acc * dacc_mag

    def _resample(self, rawNx6):
        """
        rawNx6: (N,6) -> (target_len,6) using linear interpolation in time.
        """
        N = rawNx6.shape[0]
        if N == self.target_len:
            return rawNx6.astype(np.float32)

        # old time grid: 0..N-1, new grid: 0..N-1 mapped to target_len points
        t_old = np.linspace(0.0, 1.0, N, dtype=np.float32)
        t_new = np.linspace(0.0, 1.0, self.target_len, dtype=np.float32)

        out = np.empty((self.target_len, 6), dtype=np.float32)
        for c in range(6):
            out[:, c] = np.interp(t_new, t_old, rawNx6[:, c]).astype(np.float32)
        return out

    def _zscore(self, win60x6):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not loaded. Call load_scaler(mean.npy, std.npy) first.")
        return (win60x6 - self.mean.reshape(1, 6)) / (self.std.reshape(1, 6) + 1e-6)

    def _finalize_segment(self, rawNx6):
        # Keep only last max_len samples if something went long
        if rawNx6.shape[0] > self.max_len:
            rawNx6 = rawNx6[-self.max_len:, :]

        win = self._resample(rawNx6)        # (60,6)
        win = self._zscore(win)             # (60,6)
        flat = win.reshape(-1).astype(np.float32)  # time-major flatten (360,)

        return {
            "flat360": flat,
            "window60x6": win,
            "rawNx6": rawNx6,
            "N": int(rawNx6.shape[0]),
        }