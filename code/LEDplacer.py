"""Module for placing towers on the LED strip according to a spatial Poisson
process with refractory interval (described in doi: 10.3389/fnbeh.2018.00036)
and converting those tower locations to LED indices."""

import numpy as np


class LedPlacer():
    # parameters from what we have (apparatus length and WS2812B LED strip)
    LEDS_PER_METER = 60
    APPARATUS_LENGTH = 120  # cm
    NUM_LEDS = int(APPARATUS_LENGTH * LEDS_PER_METER / 100)  # 72
    LED_SPACING = 100 / LEDS_PER_METER  # cm per LED = 1.667 cm

    # parameters from paper
    REFRACTORY_PERIOD = 12  # Minimum spacing between towers in cm

    def __init__(self,
                 rwd_density: float = 10,
                 no_rwd_density: float = 1,
                 rng: np.random.Generator | None = None):

        self.L = self.APPARATUS_LENGTH
        self.mu_reward = rwd_density
        self.mu_no_reward = no_rwd_density
        self.refractory_period = self.REFRACTORY_PERIOD
        self.rng = rng or np.random.default_rng()
        self.verify_parameters()

        self._current_reward_leds = np.array([], dtype=int)
        self._current_no_reward_leds = np.array([], dtype=int)
        self._current_reward_positions_cm = np.array([], dtype=float)
        self._current_no_reward_positions_cm = np.array([], dtype=float)

    def verify_parameters(self):
        if self.L <= 0:
            raise ValueError("L must be > 0")
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

        # 1-2) Draw n ~ Poisson(mu) that is less than the maximum
        # possible number of towers given the refractory period
        maxN = int(np.floor(self.L / self.refractory_period))
        while True:
            n = int(self.rng.poisson(mu))
            if n <= maxN:
                break

        # 3-6) Randomly distribute locations within [0, L], but
        # impose refractory interval
        Leffective = self.L - ((n - 1) * self.refractory_period)
        y = self.rng.uniform(0.0, 1.0, size=n)
        y = np.sort(y)
        y = y * Leffective + np.arange(n) * self.refractory_period

        # 7-8) Randomly rotate to get rid of edge artifacts and wrap around
        y = y + float(self.rng.uniform(0.0, self.L))
        y = np.where(y > self.L, y - self.L, y)
        return np.sort(np.round(y, rounding))

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import chisquare
    from time import time

    iters = 10000
    deltas = []
    Rs = []
    NRs = []
    weird = 0

    _Rs = []
    _NRs = []

    lp = LedPlacer(rwd_density=7.7, no_rwd_density=2.3)

    count_R = {k: 0 for k in range(lp.NUM_LEDS)}

    start_t = time()
    for i in range(iters):
        R, NR = lp.draw_towers()
        # print("Reward:", R, "No reward towers:", NR)
        delta = abs(len(R) - len(NR))

        for r in R:
            count_R[r] += 1

        if len(NR) > len(R):
            weird += 1
            _Rs.append(len(NR))
            _NRs.append(len(R))
            # R, NR = NR, R  # swap so R is always >= NR
        else:
            _Rs.append(len(R))
            _NRs.append(len(NR))

        Rs.append(len(R))
        NRs.append(len(NR))
        deltas.append(delta)

    end_t = time()
    print(f"Simulated {iters} iterations in {end_t - start_t:.2f} seconds")
    print(f"weird: {weird} out of {iters} iterations ({weird/iters:.2%})")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    bins = np.arange(0, 20, 1)

    axs[0].hist(Rs, bins=bins, density=True)
    axs[0].set_xlabel(f"#Towers REWARDED side (mu={lp.mu_reward})")
    axs[0].hist(_Rs, bins=bins, density=True, histtype='step',
                label="weird R", lw=2)

    axs[1].hist(NRs, bins=bins, density=True)
    axs[1].set_xlabel(f"#Towers NON-REWARDED side (mu={lp.mu_no_reward})")
    axs[1].hist(_NRs, bins=bins, density=True, histtype='step',
                label="weird NR", lw=2)

    axs[2].hist(deltas, bins=bins, density=True)
    axs[2].set_xlabel("Delta (|#R - #NR|)")

    for ax in axs[0:3]:
        ax.set_ylabel("Density")
        ax.set_xlim(0, 20)
        ax.set_xticks(bins)
        ax.set_ylim(0, 0.3)

    # Check distribution of reward tower placements across the LED strip
    axs[3].bar(count_R.keys(), count_R.values())
    axs[3].set_xlabel("LED Index")
    axs[3].set_ylabel("Count of Reward Towers Placed")
    axs[3].set_xlim(-1, lp.NUM_LEDS)
    axs[3].set_xticks(range(0, lp.NUM_LEDS + 1, 6))
    axs[3].set_ylim(800, 1100)

    # Chi2 test uniform
    observed = np.array(list(count_R.values()))
    expected = np.ones_like(observed, dtype=float) * observed.mean()
    axs[3].axhline(expected[0], color='red',
                   linestyle='--', label='Expected (uniform)')
    print("Expected", expected[0], iters / lp.NUM_LEDS * np.mean(Rs))

    chi2_stat, p_value = chisquare(observed, f_exp=expected)
    print(f"Chi-squared statistic: {chi2_stat:.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Reject H_0: distribution is not uniform")
    else:
        print("Fail to reject H_0: distribution is consistent with uniform")

    plt.savefig("led_placer.png")
