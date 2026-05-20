"""
!!! WIP from side project. Brunton 2018 model adapted to predict mouse decision
in the towers task, given a sequence of tower positions. Used to generate
less synthetic synthetic data for testing. Streamlined and optimized for speed
on RPi CPU, but absolutely not meant to be anything other than a quick hack
for testing.
Rest of project and benchmark code is in TowerModel/
                                            - mouse_predict.py
                                            - mouse_realtime.py
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from left_or_right import TrialSide


PARAM_NAMES = ['sigma2_i', 'B', 'lam', 'sigma2_a', 'sigma2_s',
               'phi', 'tau_phi', 'bias', 'lapse']


class DecisionMaker:
    NDELTAS = 20  # quadrature points (±5 sigma); reduced from 70, see benchmarks
    N_BINS = 21  # accumulator bins (odd); reduced from 53
    DT = 1e-2  # spatial step in metres (1 cm)

    def __init__(self, theta, L_pos, R_pos):
        (sigma2_i, B, lam, sigma2_a, sigma2_s,
         phi, tau_phi, bias, lapse) = (float(v) for v in theta)

        self._sigma2_i = sigma2_i
        self._sigma2_a = sigma2_a
        self._sigma2_s = sigma2_s
        self._lam = lam
        self._phi = phi
        self._tau_phi = tau_phi
        self._bias = bias
        self._lapse = lapse
        self.n = self.N_BINS
        self.dt = self.DT
        self.ndeltas = self.NDELTAS

        self.xc, self.dx = make_bins(B, self.n)

        # Pre-compute M0 (no-tower step) once --> O(N²) saved on ~95% of steps.
        # because it depends only on fixed params so it never needs rebuilding!
        self.M0 = transition_M(sigma2_a * self.dt, lam, 0.0,
                               self.xc, self.dx, self.n, self.dt, self.ndeltas)

        self._sL: dict[int, float] = defaultdict(float)
        self._sR: dict[int, float] = defaultdict(float)
        self.reset(L_pos, R_pos)

    def reset(self, L_pos=None, R_pos=None) -> None:
        """Reset to trial start, optionally with new tower positions.
        Passing L_pos / R_pos rebuilds the tower step-index maps so the
        same accumulator object can be reused across trials without
        re-running init.
        """
        if L_pos is not None:
            L_pos = np.sort(np.asarray(L_pos, dtype=float))
            La = (_adapt_stream(self._phi, self._tau_phi, L_pos)
                  if len(L_pos) else np.zeros(0))
            self._sL = defaultdict(float)
            for ca, p in zip(La, L_pos):
                self._sL[int(p / self.dt)] += ca
        if R_pos is not None:
            R_pos = np.sort(np.asarray(R_pos, dtype=float))
            Ra = (_adapt_stream(self._phi, self._tau_phi, R_pos)
                  if len(R_pos) else np.zeros(0))
            self._sR = defaultdict(float)
            for ca, p in zip(Ra, R_pos):
                self._sR[int(p / self.dt)] += ca
        P = np.zeros(self.n)
        P[self.n // 2] = 1.0
        M_init = transition_M(self._sigma2_i, 0.0, 0.0,
                              self.xc, self.dx, self.n, self.dt, self.ndeltas)
        self.P = M_init @ P
        self.P /= max(self.P.sum(), 1e-300)
        self._step_idx = 0

    def step(self, n: int = 1) -> float:
        """Advance n spatial steps; return P(go right) after the last step."""
        for _ in range(n):
            sL = self._sL.get(self._step_idx, 0.0)
            sR = self._sR.get(self._step_idx, 0.0)
            if sL + sR > 0.0:
                F = transition_M(
                    self._sigma2_s * (sL + sR) + self._sigma2_a * self.dt,
                    self._lam, -sL + sR,
                    self.xc, self.dx, self.n, self.dt, self.ndeltas,
                )
            else:
                F = self.M0
            self.P = F @ self.P
            self.P /= max(self.P.sum(), 1e-300)
            self._step_idx += 1
        return self.p_right()

    def p_right(self) -> float:
        """Current P(go right)."""
        xc, P, bias, n = self.xc, self.P, self._bias, self.n
        lp = int(np.clip(np.searchsorted(xc, bias, side='right') - 1,
                         0, n - 2))
        hp = lp + 1
        dd = xc[hp] - xc[lp]
        safe_dd = dd if dd > 0 else 1.0
        dh = xc[hp] - bias
        w = np.where(np.arange(n) < lp, 0.0, 1.0)
        w[hp] = 0.5 + dh / safe_dd / 2
        w[lp] = dh / safe_dd / 2
        p = max(float(np.dot(P, w)), np.finfo(float).tiny)
        return float(p * (1.0 - self._lapse) + self._lapse / 2.0)

    @property
    def position(self) -> float:
        """Current position in metres."""
        return self._step_idx * self.dt

    @staticmethod
    def positions_from_task(task, x_entry: int, x_far: int,
                            corridor_len_m: float,
                            led_dict: dict | None = None,
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (L_pos_m, R_pos_m) from LED indices to get pixel x.

        led_dict: optional {TrialSide: index_array} override; defaults to
        task._this_trial_leds.  Returns empty arrays if calibration is missing.
        """
        scale = corridor_len_m / (x_entry - x_far)
        leds = led_dict if led_dict is not None else task._this_trial_leds
        L_pos: list[float] = []
        R_pos: list[float] = []

        for side, pos_list in ((TrialSide.LEFT, L_pos),
                               (TrialSide.RIGHT, R_pos)):
            led_idx = leds[side]
            phys_x = task._to_strip_indices(led_idx, side)
            for i in phys_x:
                lp = task.led_positions.get(int(i))
                if lp is not None and lp.x_hat is not None:
                    pos_m = float(np.clip((x_entry - lp.x_hat) * scale,
                                          0.0, corridor_len_m))
                    pos_list.append(pos_m)
        return np.sort(L_pos), np.sort(R_pos)


def pinto_to_theta(pinto_params: dict | list,
                   tower_density: float = 5.0) -> np.ndarray:
    """Convert Pinto 2018 params to theta (sigma2_s per-tower)."""
    if isinstance(pinto_params, dict):
        arr = np.array([pinto_params[k] for k in PARAM_NAMES], dtype=float)
    else:
        arr = np.array(pinto_params, dtype=float)
    theta = arr.copy()
    theta[PARAM_NAMES.index('sigma2_s')] /= tower_density
    return theta


def make_bins(B: float,
              n: int = DecisionMaker.N_BINS) -> tuple[np.ndarray, float]:
    """Return (bin_centres, bin_width) for accumulator grid."""
    dx = 2.0 * B / (n - 2)
    half = (n - 1) // 2
    pos_inner = np.arange(1, half, dtype=float) * dx
    pos = np.concatenate([pos_inner, [B + dx / 2.0]])
    xc = np.concatenate([-pos[::-1], [0.0], pos])
    return xc, dx


# cache taylor coeffs to avoid recomputing on every step
_EXPM1_DIV_X_COEFFS = np.array(
    [1.0 / math.factorial(k + 1) for k in range(12)])


def expm1_div_x(x: float) -> float:
    """(exp(x)-1)/x, numerically stable near 0."""
    result = _EXPM1_DIV_X_COEFFS[-1]
    for c in _EXPM1_DIV_X_COEFFS[-2::-1]:
        result = result * x + c
    return result


def transition_M(sigma2: float, lam: float, mu: float, xc: np.ndarray,
                 dx: float, n: int = DecisionMaker.N_BINS,
                 dt: float = DecisionMaker.DT,
                 ndeltas: int = DecisionMaker.NDELTAS) -> np.ndarray:
    """(n, n) column-stochastic Fokker-Planck transition matrix."""
    delta_idx = np.arange(-ndeltas, ndeltas + 1, dtype=float)
    std = np.sqrt(abs(sigma2) + 1e-30)
    deltas = delta_idx * (5.0 * std / ndeltas)
    ps = np.exp(-0.5 * (5.0 * delta_idx / ndeltas) ** 2)
    ps /= ps.sum()

    exp_lam_dt = np.exp(lam * dt)
    edx = expm1_div_x(lam * dt)

    F = np.zeros((n, n))
    F[0, 0] = 1.0          # sticky left bound
    F[n - 1, n - 1] = 1.0  # sticky right

    for j in range(1, n - 1):
        mean_j = exp_lam_dt * xc[j] + mu * edx
        s = mean_j + deltas

        col = np.zeros(n)
        below = s <= xc[0]
        above = s >= xc[n - 1]
        interior = ~below & ~above

        col[0] += ps[below].sum()
        col[n - 1] += ps[above].sum()

        if interior.any():
            s_int = s[interior]
            p_int = ps[interior]
            raw = (s_int - xc[1]) / dx
            lp = np.clip(np.floor(raw).astype(int) + 1, 0, n - 2)
            hp = np.clip(lp + 1, 1, n - 1)
            dd = xc[hp] - xc[lp]
            safe_dd = np.where(dd > 0, dd, 1.0)
            exact = lp == hp
            w_hp = np.where(exact, 0.0, p_int * (s_int - xc[lp]) / safe_dd)
            w_lp = np.where(exact, p_int, p_int * (xc[hp] - s_int) / safe_dd)
            np.add.at(col, lp, w_lp)
            np.add.at(col, hp, w_hp)
        F[:, j] = col
    return F


def _adapt_stream(phi: float, tau_phi: float, times: np.ndarray) -> np.ndarray:
    """Compute within-stream adaptation weights."""
    n = len(times)
    if n == 0:
        return np.zeros(0)
    ici = np.diff(times)
    ca = np.empty(n)
    ca[0] = np.finfo(float).eps
    for i, dt_i in enumerate(ici):
        prod = ca[i] * phi
        log_term = (0.0 if abs(1.0 - prod) < 1e-15
                    else tau_phi * np.log(abs(1.0 - prod)))
        arg = (1.0 / tau_phi) * (-dt_i + log_term)
        ca[i + 1] = 1.0 - np.exp(arg) if prod <= 1.0 else 1.0 + np.exp(arg)
    return ca
