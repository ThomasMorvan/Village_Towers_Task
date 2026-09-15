"""Online corridor-speed estimation and thresholding."""

from collections import deque
from statistics import median

import numpy as np


CM_PER_LED = 100 / 60  # LedPicker.LED_SPACING


def px_to_cm_fit(led_positions):
    """Return a calibrated camera-x -> corridor-cm conversion.

    The corridor is zeroed at the entry. The fitted sign accounts for the
    fact that camera x decreases as the animal moves away from the entry.
    """
    calibrated = [(i, position.x_hat)
                  for i, position in led_positions.items()
                  if position.x_hat is not None and position.x_hat >= 0]
    indices, pixels = zip(*calibrated)
    cm = np.array([(i if i < 72 else 143 - i) * CM_PER_LED for i in indices],
                  dtype=float)
    pixels = np.array(pixels, dtype=float)
    slope, intercept = np.polyfit(cm, pixels, 1)

    return lambda x_px: (float(x_px) - intercept) / slope


class OnlineSpeedEstimator:
    """Estimate corridor speed and decide whether the animal is
    running too fast.

    Speed is estimated from the difference between the median positions at
    the two ends of a rolling window. Using medians makes the estimate robust
    to isolated tracking spikes.

    The gate uses hysteresis and a dwell requirement:

    - Enter FAST immediately when speed exceeds ``th_hi``.
    - Leave FAST only after ``dwell`` consecutive frames below ``th_lo``.
    - Speeds between the thresholds leave the current state unchanged.

    Missing detections are skipped. If the time between the oldest and newest
    samples exceeds ``max_gap_s``, the estimate is discarded and the gate
    returns to FAST."""

    def __init__(self, th_hi: float = 30.0, th_lo: float = 20.0,
                 dwell: int = 4, window: int = 7, ends: int = 3,
                 max_gap_s: float = 0.5):
        if ends * 2 > window:
            raise ValueError("ends*2 must be <= window")

        self.th_hi = th_hi
        self.th_lo = th_lo
        self.dwell = dwell
        self.window = window
        self.ends = ends
        self.max_gap_s = max_gap_s

        self.reset()

    def reset(self) -> None:
        """Reset the estimator to its safe state."""
        self._buf: deque = deque(maxlen=self.window)
        self.moving = True
        self.velocity: float | None = None
        self._slow_run = 0

    @property
    def speed(self) -> float | None:
        """Absolute corridor velocity in cm/s."""
        return None if self.velocity is None else abs(self.velocity)

    def update(self, t: float, pos_cm: float | None = None) -> bool:
        """Add a measurement and return whether the animal is FAST.

        ``pos_cm=None`` when detection lost -> not adding a sample."""
        if pos_cm is not None:
            self._buf.append((t, float(pos_cm)))

        if len(self._buf) < self.window:
            return self.moving

        samples = list(self._buf)
        first = samples[:self.ends]
        last = samples[-self.ends:]

        first_time = median(t for t, _ in first)
        last_time = median(t for t, _ in last)
        dt = last_time - first_time

        if dt <= 0:
            return self.moving

        if dt > self.max_gap_s:
            self.velocity = None
            self.moving = True
            self._slow_run = 0
            return self.moving

        first_pos = median(pos for _, pos in first)
        last_pos = median(pos for _, pos in last)
        self.velocity = (last_pos - first_pos) / dt

        speed = self.speed

        if speed > self.th_hi:
            self.moving = True
            self._slow_run = 0
        elif speed < self.th_lo:
            self._slow_run += 1
            if self._slow_run >= self.dwell:
                self.moving = False
        else:
            self._slow_run = 0

        return self.moving
