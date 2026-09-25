"""Fixed-N cue stimulus: n_cues cues at n_cues fixed positions, one
coin per cue. Used by TowersTaskV3 (tower_task_v3.py)."""

import numpy as np

from LEDpicker import LedPicker


class FixedCuePicker(LedPicker):
    """Stimulus with Fixed-N: n_cues cues at fixed positions, split L / R.

    The placeable corridor is cut into `n_cues` equal steps and every trial
    lights exactly one cue at each, so separation is constant and the only free
    parameter is probability `p` that any one cue goes to the rewarded side.

    `p` is the bias of the coin, not the trial's difficulty: at a fixed p the
    realised split still varies trial to trial (5/5, 7/3, 9/1, ...) because the
    coins do, and that spread is the difficulty range.
    It is set at .75 to match the Poisson generation used in V1 with mu_r=7.7
    and mu_nr=2.3 -> 7.7/(7.7+2.3) = 0.77 (theoretical; obs = 0.754)."""

    def __init__(self, n_cues: int = 10, p: float = 0.75,
                 start_dead_zone_cm: float = 10,
                 end_dead_zone_cm: float = 25,
                 rng: np.random.Generator | None = None):
        # set before super().__init__ because it calls verify_parameters()!
        self.n_cues = n_cues
        self.p = p
        self.easy = False
        super().__init__(rwd_density=0.0, no_rwd_density=0.0,
                         start_dead_zone_cm=start_dead_zone_cm,
                         end_dead_zone_cm=end_dead_zone_cm, rng=rng)

    def verify_parameters(self):
        super().verify_parameters()
        if self.n_cues < 2:
            raise ValueError("n_cues must be >= 2")
        if not 0.5 <= self.p <= 1.0:
            raise ValueError(
                f"p must be in [0.5, 1.0], got {self.p}: below 0.5 just "
                "mirrors the sides and would go against LeftOrRight draw.")
        if self.cue_separation_leds < 1:
            lo, hi = self.placeable_leds
            raise ValueError(
                f"{self.n_cues} cue positions do not fit: the dead zones "
                f"leave LED {lo} to {hi} ({hi - lo + 1} LEDs) and each cue "
                "needs its own. Lower n_cues or the dead zones.")

    @property
    def L_placeable_cm(self) -> float:
        return self.L - self.start_dead_zone_cm - self.end_dead_zone_cm

    @property
    def placeable_leds(self) -> tuple[int, int]:
        """First and last LED index a cue may use, from the dead zones."""
        return (int(np.ceil(self.start_dead_zone_cm / self.LED_SPACING)),
                min(int((self.L - self.end_dead_zone_cm) / self.LED_SPACING),
                    self.NUM_LEDS - 1))

    @property
    def cue_separation_leds(self) -> int:
        """Compute # of LEDs between consecutive cues."""
        lo, hi = self.placeable_leds
        return (hi - lo) // self.n_cues

    @property
    def cue_separation_cm(self) -> float:
        """Real spacing between consecutive cues, identical for every pair."""
        return self.cue_separation_leds * self.LED_SPACING

    @property
    def first_cue_position_led(self) -> int:
        """Half a separation past the start dead zone, so the cues do not
        begin the
        instant the animal clears it."""
        return self.placeable_leds[0] + self.cue_separation_leds // 2

    @property
    def cue_position_leds(self) -> np.ndarray:
        """LED index of each cue position: n_cues of them, evenly spaced."""
        return (self.first_cue_position_led
                + self.cue_separation_leds * np.arange(self.n_cues))

    @property
    def cue_positions_cm(self) -> np.ndarray:
        return self.cue_position_leds * self.LED_SPACING

    @property
    def end_gap_cm(self) -> float:
        """Corridor left after the last cue: end_dead_zone_cm plus whatever
        the floored separation could not use."""
        return self.L - self.cue_positions_cm[-1]

    def draw_towers(self) -> tuple[np.ndarray, np.ndarray]:
        """One Bernoulli(p) coin per cue; rewarded-side pile first."""
        leds = self.cue_position_leds
        if self.easy:
            mask = np.ones(self.n_cues, dtype=bool)
        else:
            mask = self.rng.random(self.n_cues) < self.p
        self._current_reward_leds = leds[mask]
        self._current_no_reward_leds = leds[~mask]
        self._current_reward_positions_cm = (
            self._current_reward_leds * self.LED_SPACING)
        self._current_no_reward_positions_cm = (
            self._current_no_reward_leds * self.LED_SPACING)
        return self._current_reward_leds, self._current_no_reward_leds

    def update_mu(self, rwd_density: float | None = None,
                  no_rwd_density: float | None = None):
        """Tower densities are meaningless here, but TowersTask calls this with
        no_rwd_density=0 for easy trials (warmup/rescue), so we keep it."""
        if no_rwd_density is not None:
            self.easy = no_rwd_density == 0.0  # easy trial if NR density is 0


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt
    from scipy.stats import binom

    def plot_stats(n_cues: int = 10, p: float = 0.75, iters: int = 40000,
                   save=None, seed: int = 0):
        """Same as V1 figure, but using this picker instead of Poisson."""

        lp = FixedCuePicker(n_cues=n_cues, p=p,
                            rng=np.random.default_rng(seed))
        n = lp.n_cues
        ks = np.array([len(lp.draw_towers()[0]) for _ in range(iters)])

        x = np.arange(n + 1)
        # delta = |2k - n|: with n fixed, moving one cue to other side
        # changes delta by 2. Odd n only shifts the grid to the odd values.
        dx = np.arange(n % 2, n + 1, 2)
        theory_d = np.array([binom.pmf((n + d) // 2, n, p)
                             + (binom.pmf((n - d) // 2, n, p) if d else 0.0)
                             for d in dx])
        assert np.isclose(theory_d.sum(), 1.0)

        panels = [
            # data, x grid, theory, bar width, xlabel, theory label, legend loc
            (ks, x, binom.pmf(x, n, p), 1.0, "#Cues (Rewarded)",
             f"Binomial PMF (n={n}, p={p})", "upper left"),
            (n - ks, x, binom.pmf(x, n, 1 - p), 1.0, "#Cues (Non-Rewarded)",
             f"Binomial PMF (n={n}, p={1 - p:.2f})", "upper right"),
            (np.abs(2 * ks - n), dx, theory_d, 2.0,
             "Delta (#Rewarded - #Non-Rewarded)",
             "Binomial PMF (folded)", "upper right"),
        ]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (data, grid, theory, w, xlabel, tlabel, loc) in zip(
                axs, panels):
            emp = np.array([(data == v).mean() for v in grid])
            ax.bar(grid, emp, width=0.9 * w, color="lightgray",
                   edgecolor="darkgray", label="Simulated cues drawn")
            ax.plot(grid, theory, "r--", label=tlabel)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_xticks(grid)
            ax.legend(fontsize=11, loc=loc, framealpha=1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_bounds(0, emp.max())
            ax.spines["bottom"].set_bounds(grid[0], grid[-1])
            pad = (grid[-1] - grid[0]) / 30
            ax.set_xlim(grid[0] - w / 2 - pad, grid[-1] + w / 2 + pad)
            ax.set_ylim(0, emp.max() * 1.35)
        axs[0].set_ylabel("Probability", fontsize=16)
        nr = r"$n_{NR} = n - k$"
        axs[1].text(0.97, 0.45, f"same as panel 1\nbut mirrored\n({nr})",
                    transform=axs[1].transAxes, ha="right", va="top",
                    fontsize=11, style="italic", color="dimgray")
        fig.suptitle(f"FixedCuePicker   n_cues={n}   p={p}   "
                     f"({iters} iters, cues separated by "
                     f"{lp.cue_separation_cm:.2f}cm)", fontsize=14)
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=150)
            print(f"Figure saved to {save}")
        return fig

    def plot_cue_positions(n_cues: int = 10, p: float = 0.75,
                           iters: int = 40000, save=None, seed: int = 0):
        """Where the cues are on the strip: dead_zones + LED idx."""

        lp = FixedCuePicker(n_cues=n_cues, p=p,
                            rng=np.random.default_rng(seed))
        cue_position_leds = lp.cue_position_leds
        hits = np.zeros(len(cue_position_leds))
        for _ in range(iters):
            hits += np.isin(cue_position_leds, lp.draw_towers()[0])
        frac = hits / iters

        start_led = lp.start_dead_zone_cm / lp.LED_SPACING
        end_led = (lp.L - lp.end_dead_zone_cm) / lp.LED_SPACING

        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.axvspan(0, start_led, color="gray", alpha=0.3,
                   label=f"start dead zone ({lp.start_dead_zone_cm:.0f}cm)")
        ax.axvspan(end_led, lp.NUM_LEDS, color="gray", alpha=0.3,
                   label=f"end dead zone ({lp.end_dead_zone_cm:.0f}cm)")
        # the whole-LED separation leaves a remainder at the end
        ax.axvspan(cue_position_leds[-1], end_led, color="gray", alpha=0.15,
                   label=f"separation remainder "
                         f"(end gap {lp.end_gap_cm:.1f}cm)")
        ax.bar(cue_position_leds, frac, width=1, color="lightgray",
               edgecolor="darkgray", label="P(cue on rewarded side)")
        ax.axhline(p, color="r", ls="--", label=f"p = {p}")
        for led, fr in zip(cue_position_leds, frac):
            ax.text(led, fr + 0.03, f"LED {led}", ha="center", fontsize=9)

        secax = ax.secondary_xaxis(
            "top", functions=(lambda v: v * lp.LED_SPACING,
                              lambda c: c / lp.LED_SPACING))
        secax.set_xlabel("position along corridor (cm)", fontsize=12)
        ax.set_xlim(0, lp.NUM_LEDS)
        ax.set_ylim(0, 1.4)
        ax.set_xticks(cue_position_leds)
        ax.set_xticks(np.arange(0, lp.NUM_LEDS))
        ax.set_xlabel(f"LED index (strip is 0 to {lp.NUM_LEDS - 1}, "
                      "entry end at 0)", fontsize=12)
        ax.set_ylabel("P(rewarded side)", fontsize=12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="upper center", ncol=3, framealpha=1.0)
        ax.set_title(f"n_cues={n_cues}  p={p}  cues separation "
                     f"{lp.cue_separation_cm:.2f}cm "
                     f"({lp.cue_separation_leds} LEDs) "
                     f"n_iters={iters}",
                     fontsize=13, pad=32)
        fig.tight_layout()

        print("cue positions: "
              + ", ".join(f"LED{i}@{c:.1f}cm" for i, c
                          in zip(cue_position_leds, lp.cue_positions_cm)))
        if save is not None:
            fig.savefig(save, dpi=150)
            print(f"Figure saved to {save}")
        return fig

    N, P, ITERS = 10, 0.75, 40000

    def _picker(seed: int = 0, **kw) -> FixedCuePicker:
        """A picker at the values under test, unless overridden."""
        kw.setdefault("n_cues", N)
        kw.setdefault("p", P)
        return FixedCuePicker(rng=np.random.default_rng(seed), **kw)

    def _draw_ks(lp: FixedCuePicker, iters: int = ITERS) -> np.ndarray:
        """Draw the number of rewarded cues per trial."""
        return np.array([len(lp.draw_towers()[0]) for _ in range(iters)])

    def test_cue_positions():
        """one LED per cue, identical gaps, clear of both dead zones"""
        lp = _picker()
        leds = lp.cue_position_leds
        lo, hi = lp.placeable_leds
        assert len(leds) == lp.n_cues, "one LED per cue position"
        assert len(set(np.diff(leds))) == 1, "every gap identical"
        assert np.diff(leds)[0] == lp.cue_separation_leds
        assert leds.min() >= lo, "clear of the start dead zone"
        assert leds.max() <= hi, "clear of the end dead zone"

    def test_one_cue_per_position():
        """every trial fills all n_cues positions, one side each"""
        lp = _picker()
        expected = set(lp.cue_position_leds.tolist())
        for _ in range(2000):
            rwd, no_rwd = lp.draw_towers()
            assert len(rwd) + len(no_rwd) == lp.n_cues, "N is fixed"
            assert not set(rwd) & set(no_rwd), "a cue is on one side only"
            assert set(rwd) | set(no_rwd) == expected, "same positions always"

    def test_bernoulli_per_cue():
        """k ~ Binomial(n, p), so ties and the k<n/2 tail both occur"""
        lp = _picker()
        n, p = lp.n_cues, lp.p
        ks = _draw_ks(lp)
        all_k = np.arange(n + 1)
        emp = np.array([(ks == k).mean() for k in all_k])

        # compare the empirical distribution to the theoretical distribution
        assert np.abs(emp - binom.pmf(all_k, n, p)).max() < 0.01, \
            dict(zip(all_k, emp.round(3)))
        # a count drawn first and then shuffled reproduces neither of these
        assert 0.04 < (ks == n // 2).mean() < 0.08, "5/5 ties ~5.8% at p=0.75"
        assert 0.01 < (ks < n // 2).mean() < 0.03, "k<5 tail ~2.0% at p=0.75"

    def test_majority_is_rewarded():
        """after TowersTask's swap the rewarded side holds the most cues"""
        lp = _picker()
        n = lp.n_cues
        ks = _draw_ks(lp)
        folded = np.maximum(ks, n - ks)
        assert folded.min() >= n // 2
        split = {k: (folded == k).mean() for k in range(n // 2, n + 1)}
        assert max(split, key=split.get) == 8, "8/2 is the most common split"
        print("      split " + "  ".join(f"{k}/{n - k}={v:.1%}"
                                         for k, v in split.items()))

    def test_easy_trials():
        """update_mu(_, 0) is how TowersTask asks for warmup / rescue"""
        lp = _picker()
        lp.update_mu(7.7, 0.0)
        assert all(len(lp.draw_towers()[0]) == lp.n_cues for _ in range(100))
        lp.update_mu(7.7, 2.3)
        assert any(len(lp.draw_towers()[0]) < lp.n_cues for _ in range(100))

    def test_separation_is_derived():
        """freeing corridor widens the separation, gaps stay uniform"""
        lp = _picker()
        sep = lp.cue_separation_leds
        lp.update_dead_zone(end_dead_zone_cm=0.0)
        assert lp.cue_separation_leds > sep, "free space widens separation"
        assert len(set(np.diff(lp.cue_position_leds))) == 1, "still uniform"
        assert lp.cue_position_leds[-1] <= lp.NUM_LEDS - 1, "on the strip"
        assert lp.cue_positions_cm.max() < lp.L

    def test_bad_params():
        """configurations that must be refused"""
        for bad in (dict(n_cues=80),  # more cues than placeable LEDs
                    dict(p=0.4),  # probability too low
                    dict(p=1.5),  # probability too high
                    dict(n_cues=1)):  # too few cues
            try:
                FixedCuePicker(**bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad} must be refused")

    def print_example_trials(trials: int = 5, seed: int = 0):
        """A few trials as R/L strings, easier to eyeball than index lists.
        Every trial uses the same positions, so only the sides differ."""
        lp = _picker(seed)
        leds = lp.cue_position_leds
        print("      LED  " + " ".join(f"{i:>3d}" for i in leds))
        for i in range(trials):
            m = np.isin(leds, lp.draw_towers()[0])
            print(f"      {i + 1:>3d}: "
                  + " ".join(f"  {'R' if x else 'L'}" for x in m)
                  + f"   {m.sum()}/{lp.n_cues - m.sum()}")

    def test_fixed_cue_picker():
        """Run every check above, then report."""
        for t in (test_cue_positions, test_one_cue_per_position,
                  test_bernoulli_per_cue, test_majority_is_rewarded,
                  test_easy_trials, test_separation_is_derived,
                  test_bad_params):
            t()
            print(f"  ok  {t.__name__}: {t.__doc__}")
        lp = _picker()
        print(f"OK: FixedCuePicker n_cues={lp.n_cues} p={lp.p}, cues every "
              f"{lp.cue_separation_cm:.2f}cm "
              f"({lp.cue_separation_leds} LEDs), "
              f"end gap {lp.end_gap_cm:.1f}cm")

    SAVE = Path(__file__).resolve().parent / "no_tracking" / "figures"
    test_fixed_cue_picker()
    print_example_trials()
    plot_stats(n_cues=10, p=0.75,
               save=SAVE / "fixed_cue_picker_stats.png")
    plot_cue_positions(n_cues=10, p=0.75, iters=1000,
                       save=SAVE / "fixed_cue_picker_positions.png")

    for p in (.5, .6, .7, .75, .8, .9, 1.0):
        plot_stats(n_cues=10, p=p,
                   save=SAVE / f"fixed_cue_picker_stats_{p}.png")
        plot_cue_positions(n_cues=10, p=p, iters=1000,
                           save=SAVE / f"fixed_cue_picker_positions_{p}.png")
