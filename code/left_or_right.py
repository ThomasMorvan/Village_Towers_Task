"""Implementation of the trial side selection process described in
doi: 10.3389/fnbeh.2018.00036."""

from collections import deque
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt


class TrialResult:
    def __init__(self, side: Literal["R", "L"], correct: bool):
        self.side = side  # 'R' or 'L'
        self.correct = correct  # True or False

    @classmethod
    def generate_trial(cls, side: Literal["R", "L"] | None = None,
                       correct_prob_L: float = 0.8,
                       correct_prob_R: float = 0.7
                       ) -> "TrialResult":
        if side is None:
            side = np.random.choice(['R', 'L'])
        correct_prob = correct_prob_R if side == "R" else correct_prob_L
        correct = np.random.rand() < correct_prob
        return cls(side, correct)

    def __repr__(self):
        tick = "✓" if self.correct else "✗"
        return f" {self.side} Trial {tick}"


class LeftOrRight:
    SIGMA_TRIALS = 20
    SIGMA_EMPIRICAL = 60
    MIN_RANGE = 0.15
    MAX_RANGE = 0.85
    WINDOW_SIZE = 40

    def __init__(self, verbose=True):
        self.window_size = self.WINDOW_SIZE
        self.history = deque(maxlen=self.WINDOW_SIZE)
        self.verbose = verbose

        self._cache_hg_trials = self.half_gaussian(self.WINDOW_SIZE,
                                                   self.SIGMA_TRIALS)
        self._cache_hg_empirical = self.half_gaussian(self.WINDOW_SIZE,
                                                      self.SIGMA_EMPIRICAL)

        self.PRs = []
        self.empirical_Rs = []
        self.sides = []
        self.draw_probabilities = []

    @staticmethod
    def half_gaussian(n: int, sigma: float) -> np.ndarray:
        """Half-Gaussian for n trials (most recent weighted most) with
        standard deviation sigma."""
        x = np.arange(n)
        w = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return w / w.sum()

    def add_trial(self, trial_result: TrialResult):
        self.history.append(trial_result)

    def weighted_error_fraction(self, side: str) -> float:
        """Weighted average of the fraction of errors
        in a side over the past window_size trials."""

        # get trials and sort by recent first
        trials = [t for t in self.history if t.side == side][::-1]
        if len(trials) == 0:
            return 0.5

        errors = np.array([not t.correct for t in trials], dtype=float)
        weighted = self._cache_hg_trials[:len(trials)] * errors
        if self.verbose:
            print(f"{side}: {int(np.sum(errors))}/{len(trials)} "
                  f"({100 * np.sum(errors)/len(trials):.2f}%)"
                  f" --> {np.sum(weighted):.3f}")

        return np.sum(weighted)

    def p_R(self) -> float:
        """Compute probability of drawing a right trial."""
        eR = self.weighted_error_fraction("R")
        eL = self.weighted_error_fraction("L")

        sqrt_eR = np.sqrt(eR)
        sqrt_eL = np.sqrt(eL)

        sqrt_eR = np.clip(sqrt_eR, self.MIN_RANGE, self.MAX_RANGE)  # cap
        sqrt_eL = np.clip(sqrt_eL, self.MIN_RANGE, self.MAX_RANGE)

        pR = sqrt_eR / (sqrt_eR + sqrt_eL)
        if self.verbose:
            print(f"pR: {pR:.3f} (eR={eR:.3f}, eL={eL:.3f}, "
                  f"sqrt_eR={sqrt_eR:.3f}, sqrt_eL={sqrt_eL:.3f})")

        return pR

    def empirical_fraction(self) -> float:
        """Correct pseudo-random.."""
        if len(self.history) == 0:
            return 0.5

        trials = list(self.history)[::-1]
        is_right = np.array([t.side == "R" for t in trials], dtype=float)
        weighted = self._cache_hg_empirical[:len(trials)] * is_right
        if self.verbose:
            print(f"Empirical R: {np.sum(is_right)}/{len(trials)} "
                  f"({100 * np.sum(is_right)/len(trials):.2f}%)"
                  f" --> {np.sum(weighted):.3f}")
        return np.sum(weighted)

    def draw_next_trial(self) -> str:
        """Returns 'R' or 'L' according to debiased pseudo-random rule."""
        pR = self.p_R()
        empR = self.empirical_fraction()

        if empR > pR:
            draw_prob = 0.5 * pR
        else:
            draw_prob = 0.5 * (1 + pR)

        side = "R" if np.random.rand() < draw_prob else "L"

        self.PRs.append(pR)
        self.empirical_Rs.append(empR)
        self.sides.append(side)
        self.draw_probabilities.append(draw_prob)
        return side

    def plot(self):
        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.PRs,
                label="pR (probability drawing a right trial)")
        ax.plot(self.empirical_Rs,
                label="empR (pseudo-random correction to enforce balance)")
        ax.plot(self.draw_probabilities,
                label="draw_prob (final probability used)", lw=0.5)

        ratio = []
        for i in range(len(self.sides)):
            s = self.sides[:i+1]
            r = sum(1 for t in s if t == "R") / len(s)
            ratio.append(r)
        ax.plot(ratio, label="R / (R+L)", lw=2, color='black')
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.set_xlabel("Trial")
        ax.set_ylabel("Probability")
        ax.set_xlim(0, len(self.sides))
        ax.set_ylim(0, 1)

        ax.legend()
        plt.show()


if __name__ == "__main__":
    test = LeftOrRight(verbose=False)
    sides = []
    for _ in range(1000):
        side = test.draw_next_trial()
        trial = TrialResult.generate_trial(side=side,
                                           correct_prob_L=0.8,
                                           correct_prob_R=0.8)
        test.add_trial(trial)
        sides.append(side)

    lefts = sum(1 for t in sides if t == "L")
    rights = sum(1 for t in sides if t == "R")
    print(f"10k trials: {lefts} L, {rights} R")

    test.plot()
