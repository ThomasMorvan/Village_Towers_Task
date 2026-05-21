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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
        self._B = B
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

    def plot(self, L_t, R_t, save_path=None,
             xlabel='Position (m)',
             cue_label=('L tower', 'R tower')):

        L_t = np.asarray(L_t)
        R_t = np.asarray(R_t)

        La = (_adapt_stream(self._phi, self._tau_phi, L_t) if len(L_t)
              else np.zeros(0))
        Ra = (_adapt_stream(self._phi, self._tau_phi, R_t) if len(R_t)
              else np.zeros(0))

        self.reset(L_t, R_t)

        all_pos = np.concatenate([L_t, R_t])
        max_pos = float(all_pos.max()) if len(all_pos) else 0.0
        n_steps = int(np.ceil(max(max_pos + 0.3, 1.20) / self.dt))

        xc = self.xc
        dbin = xc[1] - xc[0]
        P_trace = np.empty((n_steps + 1, self.n))
        P_trace[0] = self.P.copy()
        for i in range(n_steps):
            self.step(1)
            P_trace[i + 1] = self.P.copy()

        t_ax = np.arange(n_steps + 1) * self.dt
        mean_traj = P_trace @ xc

        fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                 gridspec_kw={'width_ratios': [3, 1],
                                              'height_ratios': [4, 1]})
        axes[1, 1].set_visible(False)

        ax = axes[0, 0]
        ext = [t_ax[0], t_ax[-1], xc[0], xc[-1]]
        im = ax.imshow(np.maximum(P_trace.T, 1e-8),
                       aspect='auto', origin='lower', extent=ext,
                       cmap='inferno', interpolation='bilinear',
                       vmin=0, vmax=np.percentile(P_trace, 99.5))
        for i, t in enumerate(R_t):
            ax.plot(t, xc[-1] * 0.88, 'v', color='lightgreen', ms=7, mec='w',
                    mew=0.5, label=cue_label[1] if i == 0 else None)
        for i, t in enumerate(L_t):
            ax.plot(t, xc[0] * 0.88, '^', color='red', ms=7, mec='w',
                    mew=0.5, label=cue_label[0] if i == 0 else None)
        ax.plot(t_ax, mean_traj, color='white', alpha=0.7,
                lw=1.2, label='E[a]')
        ax.axhline(self._bias, color='w', ls='--', alpha=0.6,
                   lw=1, label='bias')
        ax.axhline(self._B, color='cyan', ls=':', alpha=0.4, lw=1)
        ax.axhline(-self._B, color='cyan', ls=':', alpha=0.4, lw=1)
        ax.set_ylabel('a')
        ax.tick_params(labelbottom=False)
        ax.legend()
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='P(a)')

        ax = axes[0, 1]
        fd = P_trace[-1]
        cols = ['red' if b < self._bias else 'lightgreen' for b in xc]
        ax.barh(xc, fd, height=dbin, color=cols, edgecolor='none', alpha=0.75)
        ax.axhline(self._bias, color='#264653', ls='--', lw=2, label='bias')
        ax.axhline(self._B, color='#457b9d', ls=':', lw=1.2,
                   alpha=0.6, label='±B')
        ax.axhline(-self._B, color='#457b9d', ls=':', lw=1.2, alpha=0.6)
        p_right = float(np.sum(fd[xc > self._bias]))
        ax.set_xlabel('P(a, t=T)')
        ax.set_ylabel('a')
        ax.legend(fontsize=8)
        ax.text(max(fd) * 0.3, self._B * 0.6, f'P(R)≈{p_right:.3f}',
                fontsize=10, fontweight='bold')

        ax = axes[1, 0]
        ax.sharex(axes[0, 0])
        n_pts = 60
        max_pos = t_ax[-1]

        def _recovery(Ca, positions, phi, tau_phi, end_pos):
            xs, ys = [], []
            for k in range(len(positions)):
                p0 = positions[k]
                p1 = positions[k + 1] if k + 1 < len(positions) else end_pos
                state = Ca[k] * phi  # channel state right after click k
                ps = np.linspace(p0, p1, n_pts)
                s = ps - p0
                if abs(1.0 - state) < 1e-15:
                    curve = np.ones(n_pts)
                elif state <= 1.0:
                    curve = 1.0 - np.exp(-s / tau_phi) * abs(1.0 - state)
                else:
                    curve = 1.0 + np.exp(-s / tau_phi) * abs(1.0 - state)
                xs.append(ps)
                ys.append(curve)
            return np.concatenate(xs), np.concatenate(ys)

        if len(R_t) and len(Ra):
            xR, cR = _recovery(Ra, R_t, self._phi, self._tau_phi, max_pos)
            ax.plot(xR, cR, color='lightgreen', lw=1.2, label=cue_label[1])
            ax.scatter(R_t, Ra, color='lightgreen', s=35, marker='v', zorder=5)
            for t in R_t:
                ax.axvline(t, color='lightgreen', lw=0.4, alpha=0.25)

        if len(L_t) and len(La):
            xL, cL = _recovery(La, L_t, self._phi, self._tau_phi, max_pos)
            ax.plot(xL, cL, color='red', lw=1.2, label=cue_label[0])
            ax.scatter(L_t, La, color='red', s=35, marker='^', zorder=5)
            for t in L_t:
                ax.axvline(t, color='red', lw=0.4, alpha=0.25)

        ax.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Ca (adapted weight)')
        ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path is None:
            path = Path(__file__).resolve().parent.parent / 'data'
            path.mkdir(parents=True, exist_ok=True)
            save_path = path / 'plot.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved {save_path}')
        return fig

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
    ca[0] = 1  # np.finfo(float).eps  # first click bilateral in brunton,
    # but in our task we want it to have full weight
    for i, dt_i in enumerate(ici):
        prod = ca[i] * phi
        log_term = (0.0 if abs(1.0 - prod) < 1e-15
                    else tau_phi * np.log(abs(1.0 - prod)))
        arg = (1.0 / tau_phi) * (-dt_i + log_term)
        ca[i + 1] = 1.0 - np.exp(arg) if prod <= 1.0 else 1.0 + np.exp(arg)
    return ca
