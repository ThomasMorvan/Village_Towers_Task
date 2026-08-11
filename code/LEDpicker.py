"""Module for placing towers on the LED strip according to a spatial Poisson
process with refractory interval (described in doi: 10.3389/fnbeh.2018.00036)
and converting those tower locations to LED indices."""

import glob
import math
from pathlib import Path
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, iv
from scipy.stats import chisquare


def _poisson(k, mu):
    return (mu ** k) * np.exp(-mu) / factorial(k)


def _skellam(k, mu1, mu2):
    return (np.exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)
            * iv(np.abs(k), 2 * np.sqrt(mu1 * mu2)))


class LedPicker():
    # parameters from what we have (apparatus length and WS2812B LED strip)
    LEDS_PER_METER = 60
    APPARATUS_LENGTH = 120  # cm
    NUM_LEDS = int(APPARATUS_LENGTH * LEDS_PER_METER / 100)  # 72
    LED_SPACING = 100 / LEDS_PER_METER  # cm per LED = 1.667 cm

    # Minimum spacing between towers in cm
    REFRACTORY_PERIOD = 5 * LED_SPACING  # 5 LEDs (~8.33cm).

    def __init__(self,
                 rwd_density: float = 10,
                 no_rwd_density: float = 1,
                 start_dead_zone_cm: float = 0,
                 end_dead_zone_cm: float = 0,
                 refractory_period: float = REFRACTORY_PERIOD,
                 rng: np.random.Generator | None = None):

        self.L = self.APPARATUS_LENGTH
        self.mu_reward = rwd_density
        self.mu_no_reward = no_rwd_density
        self.refractory_period = refractory_period
        self.start_dead_zone_cm = start_dead_zone_cm
        self.end_dead_zone_cm = end_dead_zone_cm
        self.rng = rng or np.random.default_rng()
        self.verify_parameters()

        self._current_reward_leds = np.array([], dtype=int)
        self._current_no_reward_leds = np.array([], dtype=int)
        self._current_reward_positions_cm = np.array([], dtype=float)
        self._current_no_reward_positions_cm = np.array([], dtype=float)

    def verify_parameters(self):
        if self.L <= 0:
            raise ValueError("L must be > 0")
        if self.start_dead_zone_cm < 0:
            raise ValueError("start_dead_zone_cm must be >= 0")
        if self.end_dead_zone_cm < 0:
            raise ValueError("end_dead_zone_cm must be >= 0")
        if self.start_dead_zone_cm + self.end_dead_zone_cm >= self.L:
            raise ValueError(
                "start_dead_zone_cm + end_dead_zone_cm must be < L")
        if self.refractory_period <= 0:
            raise ValueError("refractory_period must be > 0")
        if self.mu_reward < 0:
            raise ValueError("mu_reward must be >= 0")
        if self.mu_no_reward < 0:
            raise ValueError("mu_no_reward must be >= 0")

    def place_LEDs(self, mu: float, rounding: int = 3) -> np.ndarray:
        """Spatial Poisson process with refractory interval as described
        in doi: 10.3389/fnbeh.2018.00036

        Inputs:
            L : float = Maximum possible location of the tower.
            dy : float = Minimum possible spacing between towers.
            mu : float = Tower density / mean number of towers per meter.

        Output:
            y : np.ndarray = list of locations of towers in range [0, L].
        """

        # Placement range is [start_dead_zone_cm, L - end_dead_zone_cm]
        L_placeable = self.L - self.start_dead_zone_cm - self.end_dead_zone_cm

        # 1-2) Draw n ~ Poisson(mu) that is less than the maximum
        # possible number of towers given the refractory period
        maxN = int(np.floor(L_placeable / self.refractory_period))
        n = self._draw_truncated_poisson(mu, maxN)

        # 3-6) Randomly distribute locations within [0, L_placeable], but
        # impose refractory interval.
        # see test_wrap_seam_fix: Changed from paper (n-1) -> (n) because we
        # need to reserve space between the last and first tower when wrapping
        # around, was Leffective = L_placeable-((n-1)*self.refractory_period)
        Leffective = L_placeable - (n * self.refractory_period)
        y = self.rng.uniform(0.0, 1.0, size=n)
        y = np.sort(y)
        y = y * Leffective + np.arange(n) * self.refractory_period

        # 7-8) Randomly rotate to get rid of edge artifacts and wrap around
        y = y + float(self.rng.uniform(0.0, L_placeable))
        y = np.where(y > L_placeable, y - L_placeable, y)

        # Offset into the valid placement region (past the start dead zone)
        y = y + self.start_dead_zone_cm
        return np.sort(np.round(y, rounding))

    def _draw_truncated_poisson(self, mu: float, maxN: int) -> int:
        """Draw n ~ Poisson(mu) conditioned on n <= maxN.
        This should produce the same distribution, but avoid the while loop
        when mu >> maxN, which can take forever to finish.

        The (untruncated) Poisson PMF is:
            P(N = k) = exp(-mu) * mu**k / k!,    with k = 0, 1, 2, ...

        Conditioning on N <= maxN gives:
            P(N = k | N <= maxN) = P(N = k) / P(N <= maxN)
                                 = [exp(-mu) * mu**k / k!] /
                                    sum_{j=0}^{maxN} [exp(-mu) * mu**j / j!]
        exp(-mu) cancels, so we can sample with weights proportional to
            w_k ∝ mu**k / k!,    for k = 0, 1, ..., maxN.

        We compute weights in log-space to avoid numerical overflow/underflow:
            log(w_k) = k * log(mu) - log(k!)
                     = k * log(mu) - lgamma(k + 1), since lgamma(k+1) = log(k!)

        Then we normalize and exponentiate to get the final probabilities:
            P(N = k | N <= maxN) = w_k / sum_{j=0}^{maxN} w_j

        Before exponentiating we subtract logw.max() from every log-weight:
        this shifts the largest exponent down to exp(0) = 1, avoiding
        overflow for large logw, and leaves the normalized probabilities
        unchanged since subtracting a constant in log-space is the same as
        dividing every w_k by a common factor, which cancels out in
        w / w.sum().
        """
        if mu <= 0 or maxN <= 0:
            return 0
        ks = np.arange(maxN + 1)

        # log(w_k) = k * log(mu) - lgamma(k + 1)
        logw = ks * math.log(mu) - np.array([math.lgamma(k + 1) for k in ks])
        # TODO: could use faster scipy.special.gammaln, see test_gammaln_speed.

        w = np.exp(logw - logw.max())  # shift for stability, see docstring
        return int(self.rng.choice(ks, p=w / w.sum()))

    def draw_towers(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw towers and return their LED indices."""
        reward_positions_cm = self.place_LEDs(self.mu_reward)
        no_reward_positions_cm = self.place_LEDs(self.mu_no_reward)

        reward_leds = self._cm_to_led(reward_positions_cm)
        no_reward_leds = self._cm_to_led(no_reward_positions_cm)

        self._current_reward_leds = reward_leds
        self._current_no_reward_leds = no_reward_leds
        self._current_reward_positions_cm = reward_positions_cm
        self._current_no_reward_positions_cm = no_reward_positions_cm

        return reward_leds, no_reward_leds

    def _cm_to_led(self, positions_cm: np.ndarray) -> np.ndarray:
        """Convert cm positions to nearest valid LED index, deduplicated."""
        # !!! floor instead of round to avoid edge bias
        indices = np.floor(positions_cm / self.LED_SPACING).astype(int)
        indices = np.clip(indices, 0, self.NUM_LEDS - 1)  # slightly overkill
        return np.unique(indices)  # dedup in case rounding collapses two LEDs

    def _print_current_state(self):
        print(f"RWD - cm: {self._current_reward_positions_cm}, "
              f"LEDs: {self._current_reward_leds}")
        print(f"NRWD - cm: {self._current_no_reward_positions_cm}, "
              f"LEDs: {self._current_no_reward_leds}")

    def update_mu(self, rwd_density: float | None = None,
                  no_rwd_density: float | None = None):
        """Update the mu parameters."""
        if rwd_density is not None:
            self.mu_reward = rwd_density
        if no_rwd_density is not None:
            self.mu_no_reward = no_rwd_density
        self.verify_parameters()

    def update_dead_zone(self, end_dead_zone_cm: float | None = None,
                         start_dead_zone_cm: float | None = None):
        """Update the dead-zone parameters."""
        if end_dead_zone_cm is not None:
            self.end_dead_zone_cm = end_dead_zone_cm
        if start_dead_zone_cm is not None:
            self.start_dead_zone_cm = start_dead_zone_cm
        self.verify_parameters()


def test_truncated_equals_old_reject() -> None:
    """
    The truncated-Poisson draw must:
        - match the old reject-until-n<=maxN loop distribution
        - not be stuck forever (previous `while True` freeze when mu >> maxN).
    """
    import time

    def old_reject(rng, mu, maxN):
        while True:
            n = int(rng.poisson(mu))
            if n <= maxN:
                return n

    lp = LedPicker(rng=np.random.default_rng(0))
    mu, maxN = 3.0, 10  # should work fine
    rng_old = np.random.default_rng(0)
    new = np.array([lp._draw_truncated_poisson(mu, maxN)
                    for _ in range(20000)])
    old = np.array([old_reject(rng_old, mu, maxN) for _ in range(20000)])
    assert new.max() <= maxN and new.min() >= 0
    assert abs(new.mean() - old.mean()) < 0.05, (new.mean(), old.mean())

    lp = LedPicker(rng=np.random.default_rng(1))
    t0 = time.perf_counter()
    for _ in range(10000):
        n = lp._draw_truncated_poisson(mu=50.0, maxN=1)  # accept prob ~1e-20
        assert n in (0, 1)
    assert time.perf_counter() - t0 < 1.0  # old loop: effectively forever
    print("OK: truncated Poisson matches the old distribution and can't spin")


def test_dead_zones() -> None:
    """Positions must stay within [start_dead_zone_cm, L - end_dead_zone_cm]"""
    start, end = 10, 30
    lp = LedPicker(start_dead_zone_cm=start, end_dead_zone_cm=end,
                   rng=np.random.default_rng(0))

    for _ in range(1000):
        y = lp.place_LEDs(mu=5)
        assert y.size == 0 or (y.min() >= start and y.max() <= lp.L - end)
    print(f"OK: all positions are within [{start}, {lp.L - end}] cm")


def test_wrap_seam_fix() -> None:
    """Old (reserve n-1 refractory gaps) vs new (reserve n) construction.

    Both work the same way, difference in how much of L_placeable gets set
    aside before dividing the leftover slack:
    Reserving n guarantees the wrap-seam gap (last-first gap after rotation) is
    also >= refractory_period. Previously, we reserved n-1, leaving it
    as leftover slack that shrinks as n grows, occasionally colliding two
    towers into the same LED after _cm_to_led dedup.
    """
    lp = LedPicker(rwd_density=7.7, refractory_period=8,
                   start_dead_zone_cm=10, end_dead_zone_cm=0,
                   rng=np.random.default_rng(0))
    L_placeable = lp.L - lp.start_dead_zone_cm - lp.end_dead_zone_cm
    maxN = int(L_placeable // lp.refractory_period)

    def place(n, reserve_n):
        Leffective = L_placeable - reserve_n * lp.refractory_period
        u = np.sort(lp.rng.uniform(0.0, 1.0, size=n))
        y = u * Leffective + np.arange(n) * lp.refractory_period
        y = y + float(lp.rng.uniform(0.0, L_placeable))
        return np.sort(np.where(y > L_placeable, y - L_placeable, y))

    def circular_min_gap(y):
        if len(y) < 2:
            return np.inf
        return min(np.diff(y).min(), (L_placeable - y[-1]) + y[0])

    old_min_gaps, new_min_gaps = [], []
    old_counts, new_counts = [], []
    old_dedup_collided, new_dedup_collided = [], []
    for _ in range(5000):
        n = int(lp._draw_truncated_poisson(lp.mu_reward, maxN))
        y_old = place(n, n - 1)
        y_new = place(n, n)
        old_min_gaps.append(circular_min_gap(y_old))
        new_min_gaps.append(circular_min_gap(y_new))
        old_led = lp._cm_to_led(y_old)
        new_led = lp._cm_to_led(y_new)
        old_counts.append(len(old_led))
        new_counts.append(len(new_led))
        old_dedup_collided.append(len(old_led) < len(y_old))
        new_dedup_collided.append(len(new_led) < len(y_new))

    old_min_gaps, new_min_gaps = np.array(old_min_gaps), np.array(new_min_gaps)
    old_counts, new_counts = np.array(old_counts), np.array(new_counts)
    old_dedup_collided = np.array(old_dedup_collided)
    new_dedup_collided = np.array(new_dedup_collided)
    rp = lp.refractory_period

    assert (new_min_gaps >= rp - 1e-9).all(), \
        "reserving n gaps must guarantee refractory spacing at the wrap seam"
    assert (old_min_gaps < rp - 1e-9).any(), \
        "reserving n-1 gaps should occasionally violate it at the wrap seam"
    assert not new_dedup_collided.any(), \
        "reserving n gaps must never let _cm_to_led's dedup drop a tower"
    assert old_dedup_collided.any(), \
        "reserving n-1 gaps may lose a tower to _cm_to_led dedup"
    assert new_counts.mean() > old_counts.mean(), \
        "n-1 should undercount relative to n (lost towers to collisions)"

    old_violation_rate = (old_min_gaps < rp - 1e-9).mean()
    new_violation_rate = (new_min_gaps < rp - 1e-9).mean()
    print(f"old (n-1): wrap-seam violated {old_violation_rate:.1%} of draws, "
          f"_cm_to_led dedup actually dropped a tower in "
          f"{old_dedup_collided.mean():.1%} of draws, "
          f"mean count = {old_counts.mean():.3f}")
    print(f"new (n):   wrap-seam violated {new_violation_rate:.1%} of draws, "
          f"_cm_to_led dedup dropped a tower in "
          f"{new_dedup_collided.mean():.1%} of draws, "
          f"mean count = {new_counts.mean():.3f}")


def test_gammaln() -> None:
    """math.lgamma (loop) vs scipy.special.gammaln (vectorized) same result,
    faster only past ~50, at the cost of scipy import in task, so nah."""
    import timeit
    from scipy.special import gammaln

    def old(maxN):
        ks = np.arange(maxN + 1)
        return np.array([math.lgamma(k + 1) for k in ks])

    def new(maxN):
        ks = np.arange(maxN + 1)
        return gammaln(ks + 1)

    for maxN in (1, 5, 10, 20, 50, 100, 1000):
        assert np.allclose(old(maxN), new(maxN))
        t_old = timeit.timeit(lambda: old(maxN), number=2000)
        t_new = timeit.timeit(lambda: new(maxN), number=2000)
        print(f"maxN={maxN:5d}: lgamma loop={t_old * 1e6 / 2000:8.2f} us  "
              f"gammaln vec={t_new * 1e6 / 2000:8.2f} us  "
              f"speedup={t_old / t_new:5.2f}x")


def _leds(s: str) -> np.ndarray:
    return np.fromstring(s[1:-1], sep=",", dtype=int)


def load(path: Path) -> pd.DataFrame:
    files = [f for f in glob.glob(str(path / "*/*_TowersTask_*.csv"))
             if "RAW" not in f]
    df = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)
    df = df[(df.stage == 5) & (df.phase == "main") & df.trial_correct.notna()]
    df = df.reset_index(drop=True)

    df["choice"] = ((df.trial_side == "R") == df.trial_correct).astype(int)
    L = df["L LEDs"].map(_leds)
    R = df["R LEDs"].map(_leds)
    rewarded_R = (df.trial_side == "R").to_numpy()
    df["cues_rewarded"] = np.where(rewarded_R, R, L)
    df["cues_unrewarded"] = np.where(rewarded_R, L, R)

    dead_zone = (10, 0)
    if (not df.empty
            and "led_start_dead_zone_cm" in df.columns
            and "led_end_dead_zone_cm" in df.columns):  # get deadzone from df
        dead_zone = (df["led_start_dead_zone_cm"].mode().iloc[0],
                     df["led_end_dead_zone_cm"].mode().iloc[0])
    if not df.empty:
        print(f"Subjects: {df['subject'].value_counts().to_dict()}")
    df.attrs = dict(label_R="REWARDED", label_nR="UNREWARDED",
                    mu_R=7.7, mu_nR=2.3,
                    dead_zone=dead_zone)
    return df


def simulate_trials(lp: LedPicker, iters: int = 10000,
                    verbose: bool = False) -> pd.DataFrame:
    t0 = time()
    rows = [{"cues_rewarded": R, "cues_unrewarded": NR}
            for R, NR in (lp.draw_towers() for _ in range(iters))]
    if verbose:
        print(f"Simulated {iters} iterations in {time() - t0:.2f} seconds")

    df = pd.DataFrame(rows)
    df.attrs = dict(
        label_R=f"SIMUL REWARDED (mu={lp.mu_reward})",
        label_nR=f"SIMUL NON-REWARDED (mu={lp.mu_no_reward})",
        mu_R=lp.mu_reward, mu_nR=lp.mu_no_reward,
        dead_zone=(lp.start_dead_zone_cm, lp.end_dead_zone_cm))
    return df


def plot_stats(df: pd.DataFrame) -> plt.Figure | None:
    if df.empty:
        print("No data to plot.")
        return None

    color = "gray"
    if df.attrs.get("label_R", "R").startswith("SIMUL REWARDED"):
        color = "blue"
    label_R = df.attrs.get("label_R", "R")
    label_nR = df.attrs.get("label_nR", "nR")
    start_cm, end_cm = df.attrs.get("dead_zone", (0, 0))
    start_led = start_cm / LedPicker.LED_SPACING
    end_led = end_cm / LedPicker.LED_SPACING

    n_R = df["cues_rewarded"].apply(len).to_numpy()
    n_nR = df["cues_unrewarded"].apply(len).to_numpy()
    deltas = np.abs(n_R - n_nR)

    mu_R = df.attrs.get("mu_R", n_R.mean())
    mu_nR = df.attrs.get("mu_nR", n_nR.mean())
    positions = np.concatenate([*df["cues_rewarded"], *df["cues_unrewarded"]])

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0])]
    gs_br = gs[1, 1].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
    axs.append(fig.add_subplot(gs_br[0]))
    ax3z = fig.add_subplot(gs_br[1], sharex=axs[3])
    axs[3].tick_params(labelbottom=False)

    nmax = max(n_R.max(), n_nR.max(), 1)
    bins = np.arange(-0.5, nmax + 1.5, 1)
    x = np.arange(0, nmax + 1)

    axs[0].hist(n_R, bins=bins, density=True, color=color)
    axs[0].plot(x, _poisson(x, mu_R), 'r--',
                label=f"Poisson PDF (mu={mu_R:.2f})")
    axs[0].set_xlabel(f"#Towers {label_R} (empirical mean={n_R.mean():.2f})")

    axs[1].hist(n_nR, bins=bins, density=True, color=color)
    axs[1].plot(x, _poisson(x, mu_nR), 'r--',
                label=f"Poisson PDF (mu={mu_nR:.2f})")
    axs[1].set_xlabel(f"#Towers {label_nR} (empirical mean={n_nR.mean():.2f})")

    dbins = np.arange(deltas.min() - 0.5, deltas.max() + 1.5, 1)
    dx = np.arange(deltas.min(), deltas.max() + 1)
    axs[2].hist(deltas, bins=dbins, density=True, color=color)
    axs[2].plot(dx, _skellam(dx, mu_R, mu_nR), 'r--', label="Skellam PDF")
    axs[2].set_xlabel(f"Delta (#{label_R} - #{label_nR})")

    for ax in axs[:3]:
        ax.set_ylabel("Density")
        ax.legend()

    pos_bins = np.arange(0, LedPicker.NUM_LEDS + 1, 1)
    counts, edges, _ = axs[3].hist(positions, bins=pos_bins, color=color)
    if start_led or end_led:
        axs[3].axvspan(0, start_led, color='gray', alpha=0.3,
                       label='Start Dead Zone')
        axs[3].axvspan(LedPicker.NUM_LEDS - end_led,
                       LedPicker.NUM_LEDS, color='gray', alpha=0.3,
                       label='End Dead Zone')
        axs[3].legend()
    axs[3].set_ylabel("Count of towers placed")
    axs[3].set_ylim(counts.max() * 0.8, counts.max() * 1.1)
    axs[3].set_xlim(0, LedPicker.NUM_LEDS)

    centers = (edges[:-1] + edges[1:]) / 2
    mask = (centers >= start_led) & (centers <= LedPicker.NUM_LEDS - end_led)
    observed = counts[mask]
    if observed.sum() > 0:
        expected = np.full_like(observed, observed.mean())
        axs[3].plot(centers[mask], expected, 'r--')
        chi2_stat, p_value = chisquare(observed, f_exp=expected)
        print(f"Chi-squared statistic: {chi2_stat:.2f}, "
              f"p-value: {p_value:.4f}")
        print("Reject H_0: distribution is not uniform"
              if p_value < 0.05 else
              "Fail to reject H_0: distribution is consistent with uniform")

        # Per-bin deviation from uniform
        z = (observed - expected) / np.sqrt(expected)
        ax3z.bar(centers[mask], z, width=1.0,
                 color=np.where(np.abs(z) > 2, 'red', 'black'))
        ax3z.axhline(2, color='gray', ls=':', lw=1)
        ax3z.axhline(-2, color='gray', ls=':', lw=1)
        ax3z.set_ylabel("z-score")
        ax3z.set_xlabel("LED index")
    else:
        ax3z.set_xlabel("LED index")

    fig.tight_layout()
    return fig


def analyze_cue_duration_fit(
        refractory_period: float = 5 * LedPicker.LED_SPACING,
        end_dead_zone_cm: float = 25,
        led_ms: float = 200,
        optimal_led_ms: float = 100,
        ns: tuple = (3, 5, 7, 9)) -> plt.Figure:
    """Analysis to check which LED_ms fit with the new refractory_period
    and end_dead_zone config. Overlap depends only on refractory_period and
    animal running speed.
    Will need cleaning later, lazy import code/no_tracking/speed.py.
    """
    import glob
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "no_tracking"))
    # FIXME: lazy import for debugging, need to clean up later
    import speed  # noqa: E402

    mice = sorted(os.path.basename(p) for p in glob.glob("data/videos/*")
                  if os.path.isdir(p))
    all_speeds = np.concatenate([speed.run_speed_samples(m) for m in mice])
    gap_ms = 1000 * refractory_period / all_speeds

    overlap_now = 100 * np.mean(gap_ms < led_ms)
    print(f"candidate: refractory={refractory_period:.2f}cm, "
          f"end_dead_zone={end_dead_zone_cm}cm, led_ms={led_ms:.0f}ms "
          f"-> predicted overlap = {overlap_now:.1f}%")

    print("led_ms needed to hit a target overlap:")
    targets = (1, 5, 10, 20)
    needed_ms = {t: float(np.percentile(gap_ms, t)) for t in targets}
    for t in targets:
        print(f"  overlap<={t:2d}% needs led_ms <= {needed_ms[t]:5.1f} ms")

    # cross-check the model against real experimental data
    prod_refractory = LedPicker.REFRACTORY_PERIOD
    prod_gap_ms = 1000 * prod_refractory / all_speeds
    model_overlap_prod = 100 * np.mean(prod_gap_ms < led_ms)

    events = speed.load_all_led_events("data/sessions", stage=5, phase="main")
    l_iti = speed.cue_intervals(events, within_trial=True, side="L")
    r_iti = speed.cue_intervals(events, within_trial=True, side="R")
    real_iti_ms = 1000 * np.concatenate([l_iti, r_iti])
    real_overlap_prod = 100 * np.mean(real_iti_ms < led_ms)

    print(f"\ncross-check @ production refractory={prod_refractory}cm: "
          f"model={model_overlap_prod:.1f}% vs real experimental="
          f"{real_overlap_prod:.1f}%")

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.ravel()

    # A) candidate gap_ms distribution
    ax = axs[0]
    edges = np.linspace(0, np.percentile(gap_ms, 99), 60)
    ax.hist(gap_ms, bins=edges, color="tab:orange", alpha=0.85)
    ax.axvline(led_ms, color="crimson", lw=1.5, ls=":",
               label=f"led_ms={led_ms:.0f}ms (overlap={overlap_now:.0f}%)")
    ax.axvline(needed_ms[5], color="k", lw=1.5, ls="--",
               label=f"led_ms for <=5% overlap = {needed_ms[5]:.0f}ms")
    ax.set_xlabel("Predicted inter-cue gap (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Candidate: refractory={refractory_period:.2f}cm "
                 f"({round(refractory_period / LedPicker.LED_SPACING)} LEDs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # B) overlap% vs led_ms sweep, at the candidate refractory
    ax = axs[1]
    led_ms_sweep = np.linspace(10, 250, 200)
    overlap_sweep = [100 * np.mean(gap_ms < m) for m in led_ms_sweep]
    ax.plot(led_ms_sweep, overlap_sweep, color="tab:orange")
    ax.axvline(led_ms, color="crimson", lw=1.5, ls=":",
               label=f"current={led_ms:.0f}ms")
    for t, color in zip(targets, ("green", "olive", "goldenrod", "firebrick")):
        ax.axhline(t, color=color, lw=1, ls=":", alpha=0.6)
        ax.axvline(needed_ms[t], color=color, lw=1, ls=":", alpha=0.6,
                   label=f"<={t}%: led_ms<={needed_ms[t]:.0f}ms")
    ax.set_xlabel("led_ms (cue duration)")
    ax.set_ylabel("Predicted overlap (%)")
    ax.set_title("How much led_ms needs to shrink")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # C) model vs real, at production refractory
    ax = axs[2]
    edges = np.linspace(0, np.percentile(real_iti_ms, 99), 60)
    ax.hist(real_iti_ms, bins=edges, color="tab:blue", alpha=0.6,
            density=True, label=f"real (overlap={real_overlap_prod:.0f}%)")
    ax.hist(prod_gap_ms, bins=edges, color="tab:orange", alpha=0.5,
            density=True, label=f"model (overlap={model_overlap_prod:.0f}%)")
    ax.axvline(led_ms, color="crimson", lw=1.5, ls=":",
               label=f"led_ms={led_ms:.0f}ms")
    ax.set_xlabel("Inter-cue gap (ms)")
    ax.set_ylabel("Density")
    ax.set_title(f"Model vs real @ refractory={prod_refractory}cm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # D) with the optimal_led_ms fix applied, at the candidate refractory
    ax = axs[3]
    overlap_opt = 100 * np.mean(gap_ms < optimal_led_ms)
    ax.hist(gap_ms, bins=edges, color="tab:orange", alpha=0.85)
    ax.axvline(led_ms, color="crimson", lw=1.5, ls=":",
               label=f"led_ms={led_ms:.0f}ms (overlap={overlap_now:.0f}%)")
    lbl = f"led_ms={optimal_led_ms:.0f}ms (overlap={overlap_opt:.0f}%)"
    ax.axvline(optimal_led_ms, color="tab:green", lw=2, ls="-", label=lbl)
    ax.set_xlabel("Predicted inter-cue gap (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"With the fix: led_ms={optimal_led_ms:.0f}ms "
                 f"@ refractory={refractory_period:.2f}cm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # E) real placement spacing (not just the refractory floor) for
    # trials with n=3,5,7,9 towers - more towers packed into the same
    # L_placeable means tighter average spacing, so overlap should worsen
    # as n grows.
    start_dead_zone_cm = 10
    L_placeable = (LedPicker.APPARATUS_LENGTH
                   - start_dead_zone_cm
                   - end_dead_zone_cm)
    rng = np.random.default_rng(0)

    def _place_n_towers(n, iters=3000):
        """Same placement math as LedPicker.place_LEDs (steps 3-8), forcing
        an exact tower count instead of drawing it from the truncated
        Poisson, so we can look at "trials with exactly n towers" directly."""
        Leffective = L_placeable - n * refractory_period
        u = rng.uniform(0.0, 1.0, size=(iters, n))
        y = np.sort(u, axis=1) * Leffective + np.arange(n) * refractory_period
        y = y + rng.uniform(0.0, L_placeable, size=(iters, 1))
        y = np.where(y > L_placeable, y - L_placeable, y)
        y = np.sort(y, axis=1)
        return np.diff(y, axis=1).ravel()  # n-1 forward gaps per trial, cm

    ax = axs[4]
    n_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ns)))
    overlap_by_n_current, overlap_by_n_opt = {}, {}
    for n, color in zip(ns, n_colors):
        cm_gaps = _place_n_towers(n)
        speeds_paired = rng.choice(all_speeds, size=len(cm_gaps), replace=True)
        gap_ms_n = 1000 * cm_gaps / speeds_paired
        overlap_by_n_current[n] = 100 * np.mean(gap_ms_n < led_ms)
        overlap_by_n_opt[n] = 100 * np.mean(gap_ms_n < optimal_led_ms)

        lbl = f"n={n} (overlap@{led_ms:.0f}ms={overlap_by_n_current[n]:.0f}%)"
        ax.hist(gap_ms_n, bins=np.linspace(0, 800, 60), histtype="step",
                lw=1.8, color=color, label=lbl)
    ax.axvline(led_ms, color="crimson", lw=1.5, ls=":",
               label=f"led_ms={led_ms:.0f}ms")
    ax.axvline(optimal_led_ms, color="tab:green", lw=1.5, ls="-",
               label=f"led_ms={optimal_led_ms:.0f}ms")
    ax.set_xlabel("Inter-cue gap (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Real placement spacing by trial tower-count "
                 f"(refractory={refractory_period:.2f}cm)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # F) overlap% by n, current vs optimal led_ms
    ax = axs[5]
    x = np.arange(len(ns))
    w = 0.35
    ax.bar(x - w / 2, [overlap_by_n_current[n] for n in ns], width=w,
           color="crimson", alpha=0.8, label=f"led_ms={led_ms:.0f}ms")
    ax.bar(x + w / 2, [overlap_by_n_opt[n] for n in ns], width=w,
           color="tab:green", alpha=0.8,
           label=f"led_ms={optimal_led_ms:.0f}ms")
    ax.set_xticks(x, labels=[str(n) for n in ns])
    ax.set_xlabel("Towers in trial (n)")
    ax.set_ylabel("Predicted overlap (%)")
    ax.set_title("Overlap by trial tower-count, before/after the fix")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    figs_dir = Path(__file__).resolve().parent / "no_tracking" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    save_path = figs_dir / "led_picker_cue_duration_fit.png"
    fig.savefig(save_path)
    print(f"\nSaved to {save_path}")
    return fig


if __name__ == "__main__":
    DATA = Path(__file__).resolve().parent.parent / "data" / "sessions"
    SAVE = Path(__file__).resolve().parent / "no_tracking" / "figures"

    lp = LedPicker(rwd_density=7.7, no_rwd_density=2.3,
                   start_dead_zone_cm=10, end_dead_zone_cm=0,
                   refractory_period=12)

    print("=" * 60)
    print("Verify implementation matches the paper's model and "
          "visualize the distributions.")
    print("=" * 60)
    print(lp.draw_towers())
    simulated = simulate_trials(lp, iters=10000)
    fig = plot_stats(simulated)
    if fig is not None:
        fig.savefig(SAVE / "led_picker_simulated.png")

    print("\nChecks on found edge cases")
    print("Test truncated Poisson (matches paper method + no hang)...")
    test_truncated_equals_old_reject()
    print("Test dead zones (positions stay within bounds)...")
    test_dead_zones()
    print("Test wrap-seam fix (reserving n vs n-1 refractory gaps)...")
    test_wrap_seam_fix()
    print("Test lgamma vs gammaln (perf only, not worth the extra import)...")
    test_gammaln()

    print("Sanity check on real experimental data.")
    exp_data = load(DATA)
    exp_fig = plot_stats(exp_data)
    if exp_fig is not None:
        exp_fig.savefig(SAVE / "led_picker_experimental.png")

    print("\nSearching for optimal end dead-zone and refractory period..."
          "We want at least 20-30cm of deadzone, and to fit as many cues, we "
          "need a shorter refractory period. But not too short or the animal "
          "will see overlapping cues, while maintaining non-pathological "
          "distributions (mu_ratio and %trials with |Δ| >= 10 are good proxies"
          " to identify if distributions are clamped). So grid-searching for "
          "something between 20-30cm deadzone and 5-10cm refractory period "
          "that still has some trials with |Δ| >= 10 and mu_ratio ~1.0.")

    def proportion_delta_gt_10(df: pd.DataFrame) -> float:
        """Proportion of trials where |Δ| towers is greater than 10."""
        deltas = np.abs(df["cues_rewarded"].apply(len)
                        - df["cues_unrewarded"].apply(len))
        return (deltas >= 10).mean()

    def mu_ratio(df: pd.DataFrame, mu_R: float, mu_nR: float
                 ) -> tuple[float, float]:
        """Empirical mean tower count as a fraction of nominal mu, per side -
        how truncated each side is (1.0 = no truncation, < 1.0 = we lost some
        towers because no space to place them)."""
        ratio_R = df["cues_rewarded"].apply(len).mean() / mu_R
        ratio_nR = df["cues_unrewarded"].apply(len).mean() / mu_nR
        return ratio_R, ratio_nR

    print(f"-> {100 * proportion_delta_gt_10(simulated):.1f}% of trials have "
          f"|delta towers| >= 10 at production settings (end dead_zone=0, "
          f"refractory={LedPicker.REFRACTORY_PERIOD}cm).")

    fig, ax = plt.subplots(figsize=(10, 8))

    mu_R, mu_nR = 7.7, 2.3
    deadzones_leds = np.arange(0, 31, 1)
    deadzones = deadzones_leds * LedPicker.LED_SPACING  # sweep with led idx
    refracts_leds = np.arange(2, 10, 1)
    refracts = refracts_leds * LedPicker.LED_SPACING  # sweep with led idx

    results_pickle = SAVE / "led_picker_heatmap_results.npy"
    if results_pickle.exists():
        print(f"Loading cached results from {results_pickle}")
        results = np.load(results_pickle, allow_pickle=True)
    else:
        results = []
        start = time()
        for dz in deadzones:
            print(f"Simulating for end dead zone {dz:.2f} cm...")
            for rp in refracts:
                lp = LedPicker(rwd_density=mu_R, no_rwd_density=mu_nR,
                               start_dead_zone_cm=10, end_dead_zone_cm=dz,
                               refractory_period=rp)
                simulated = simulate_trials(lp, iters=10000, verbose=False)
                prop = proportion_delta_gt_10(simulated)
                ratio_R, ratio_nR = mu_ratio(simulated, mu_R, mu_nR)
                results.append((dz, rp, ratio_R, ratio_nR, prop))
        results = np.array(results)
        np.save(results_pickle, results)
        print(f"Simulations completed in {time() - start:.2f} seconds")

    dz_vals = results[:, 0]
    rp_vals = results[:, 1]
    ratioR_vals = results[:, 2]
    ratioNR_vals = results[:, 3]
    prop_vals = results[:, 4]

    heatmap_data = prop_vals.reshape(len(deadzones), len(refracts))
    ratioR_grid = ratioR_vals.reshape(len(deadzones), len(refracts))
    rp_step = refracts[1] - refracts[0]
    dz_step = deadzones[1] - deadzones[0]
    extent = [refracts[0] - rp_step / 2, refracts[-1] + rp_step / 2,
              deadzones[0] - dz_step / 2, deadzones[-1] + dz_step / 2]

    c = ax.imshow(heatmap_data, origin='lower', cmap='viridis',
                  extent=extent, aspect='auto')
    xtick_labels = [f"{n}\n{n * LedPicker.LED_SPACING:.2f} cm"
                    for n in refracts_leds]
    ytick_labels = [f"{n} ({n * LedPicker.LED_SPACING:.0f} cm)"
                    for n in deadzones_leds]
    ax.set_xticks(refracts, labels=xtick_labels)
    ax.set_yticks(deadzones, labels=ytick_labels)
    fig.colorbar(c, ax=ax, label='Proportion of trials with |Δ| >= 10')

    # Truncation contours for R side
    cs = ax.contour(refracts, deadzones, ratioR_grid,
                    levels=[0.7, 0.85, 0.9, 0.95, .98, .99],
                    colors=['red', 'orange', 'green',
                            'blue', 'purple', 'brown'], zorder=6)
    ax.clabel(cs, fmt={0.7: '70% of mu_R', 0.85: '85% of mu_R',
                       0.9: '90% of mu_R', 0.95: '95% of mu_R',
                       0.98: '98% of mu_R', 0.99: '99% of mu_R'})

    # Target deadzone range for the reverse-correlation fix (20-30cm).
    ax.axhspan(20, 30, color='white', alpha=0.15)

    # optimal?
    ax.plot(5 * LedPicker.LED_SPACING, 25, marker='*', markersize=12,
            markerfacecolor='k', markeredgecolor='k', zorder=5)

    ax.set_xlabel('Refractory period (LEDs)')
    ax.set_ylabel('End dead zone (LEDs)')

    in_range = (dz_vals >= 20) & (dz_vals <= 30) & (ratioR_vals >= 0.85)
    if in_range.any():
        cand = np.column_stack([dz_vals, rp_vals, ratioR_vals, ratioNR_vals,
                                prop_vals])[in_range]
        cand = cand[cand[:, 0].argsort()]
        print("Candidates with dead_zone in [20,30] and rewarded mean "
              ">=85% of nominal mu_R:")
        for dz, rp, rR, rNR, prop in cand:
            dz_led = round(dz / LedPicker.LED_SPACING)
            rp_led = round(rp / LedPicker.LED_SPACING)
            print(f"  dead_zone={dz_led} LEDs ({dz:.2f}cm) "
                  f"refractory={rp_led} LEDs ({rp:.2f}cm): "
                  f"rewarded={100*rR:.0f}% nominal, "
                  f"unrewarded={100*rNR:.0f}% nominal, "
                  f"P(|delta|>=10)={100*prop:.1f}%")
    else:
        print("No combo in [20,30] cm reaches 85% of nominal mu_R - "
              "try smaller refractory_period.")
    plt.savefig(SAVE / "led_picker_heatmap.png")

    print("\nIsolate each parameter's effect by slicing the grid "
          "at the leading candidate, dead_zone=25cm (15 LEDs) and "
          "refractory=8.33cm (5 LEDs), holding the other one fixed.")
    fig2, axs2 = plt.subplots(2, 2, figsize=(11, 8))

    dz_fixed = 25.0  # cm (15 LEDs)
    rp_fixed = 5 * LedPicker.LED_SPACING  # cm (5 LEDs, 8.33cm)
    dz_leds = round(dz_fixed / LedPicker.LED_SPACING)

    def _slice(fixed_vals, fixed_at, sweep_vals):
        mask = np.isclose(fixed_vals, fixed_at)
        order = np.argsort(sweep_vals[mask])
        leds = np.round(sweep_vals[mask][order]
                        / LedPicker.LED_SPACING).astype(int)
        return leds, order, mask

    def _plot_delta(ax, x, y_pct, zero_thresh=0.05):
        ax.plot(x, y_pct, '-', color='darkorange')
        near_zero = y_pct <= zero_thresh
        ax.scatter(x[~near_zero], y_pct[~near_zero], marker='o',
                   color='darkorange')
        ax.scatter(x[near_zero], y_pct[near_zero], marker='x',
                   color='darkorange', s=60, zorder=3)

    rp_leds, order, mask = _slice(dz_vals, dz_fixed, rp_vals)
    axs2[0, 0].plot(rp_leds, 100 * ratioR_vals[mask][order], 'o-')
    axs2[0, 0].axhline(100, color='gray', ls=':', lw=1)
    axs2[0, 0].set_ylabel('Rewarded mean, % of nominal mu_R')
    axs2[0, 0].set_title(f'Fixed dead_zone = {dz_fixed}cm ({dz_leds} LEDs)')

    _plot_delta(axs2[1, 0], rp_leds, 100 * prop_vals[mask][order])
    axs2[1, 0].set_xlabel('Refractory period (LEDs)')
    axs2[1, 0].set_ylabel('P(|delta| >= 10), %')

    dz_leds, order, mask = _slice(rp_vals, rp_fixed, dz_vals)
    axs2[0, 1].plot(dz_leds, 100 * ratioR_vals[mask][order], 'o-')
    axs2[0, 1].axhline(100, color='gray', ls=':', lw=1)
    axs2[0, 1].axvspan(12, 18, color='green', alpha=0.1)  # 20-30cm target
    rp = int(rp_fixed / LedPicker.LED_SPACING)
    axs2[0, 1].set_title(f'Fixed refractory = {rp} LEDs ({rp_fixed:.2f}cm)')

    _plot_delta(axs2[1, 1], dz_leds, 100 * prop_vals[mask][order])
    axs2[1, 1].axvspan(12, 18, color='green', alpha=0.1)
    axs2[1, 1].set_xlabel('End dead zone (LEDs)')

    fig2.tight_layout()
    fig2.savefig(SAVE / "led_picker_slices.png")
    print(f"Saved slice plots to {SAVE / 'led_picker_slices.png'}")

    print("\nSanity check the new distribution with the leading new params")
    lp_opt = LedPicker(rwd_density=mu_R, no_rwd_density=mu_nR,
                       start_dead_zone_cm=10, end_dead_zone_cm=25,
                       refractory_period=5 * LedPicker.LED_SPACING)
    opt_fig = plot_stats(simulate_trials(lp_opt, iters=10000))
    if opt_fig is not None:
        opt_fig.savefig(SAVE / "led_picker_optimal.png")
        print(f"Saved to {SAVE / 'led_picker_optimal.png'}")

    print("\nTest new refractory period (8.33cm) leaves room for the current "
          "cue duration (led_ms=200ms) given how fast mice actually run "
          "(no adjacent cues overlapping). Cross-checked against real "
          "led_on_times timestamps.")
    try:
        analyze_cue_duration_fit()
    except Exception as e:
        print(f"  skipped (no tracking/speed data available): {e!r}")

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS:")
    print("=" * 60)
    print("""New target parameters:
  start_dead_zone_cm = 10
  end_dead_zone_cm   = 25 (15 LEDs)
  refractory_period  = 8.33 (5 LEDs)
  led_ms target = 100ms (down from 200ms)
""")
