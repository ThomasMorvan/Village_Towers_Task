import numpy as np
from dataclasses import dataclass

# TODO: add a class Staircase to encapsulate the staircase logic and
# parameters, and move the compute_step method there. This would clean up the
# StageConfig and make the code more modular.
# TODO: Big refactor, I don't like the fact that Task perfoms checkpoint and
# stage advancement logic. Maybe have a separate StageManager class that
# handles stage transitions and checkpoints, and the Task just updates the
# manager and queries it for the current difficulty parameters and whether
# to advance or not. I would feel better about it.


@dataclass
class Difficulty:
    """Active difficulty parameters for one trial."""
    mu_r: float = 0.0  # rewarded tower density (m^-1)
    mu_nr: float = 0.0  # non-rewarded tower density (m^-1)
    led_ms: int = 5000  # LED on-duration (ms)


@dataclass(frozen=True)
class StageConfig:
    stage: int
    name: str
    rwd_density: float
    no_rwd_density: float  # starting value; staircase overrides in s2
    trial_is_cued: bool  # always False in v2
    give_free_reward: bool  # always False in v2
    both_sides_rewarded: bool  # True only in Stage 0
    staircase_variable: str  # "none" or "minority_density" or "tower_duration"
    staircase_start: float  # initial difficulty at stage entry
    staircase_target: float  # value to get to (eg mu_nr=2.0)
    staircase_harder_direction: str  # "up" (mu_nr) or "down" (led_ms)
    color: str = "w"  # plot color for this stage
    advance_threshold: float = 0.70  # rolling accuracy needed to advance
    advance_label: str = ""  # string to show on hud
    timed_leds: bool = False  # T: LEDs flash for led_ms then off; F: stay on
    has_warmup: bool = False  # True: session starts with easy one-sided warmup

    def compute_step(self, correct: bool, streak: int,
                     mult_trial: int, settings) -> tuple[float, int]:
        """Compute staircase step and updated streak for one trial.

        Returns (delta, new_streak).
        mult_trial: trial count within the main phase (1-based). Pass
        0 to suppress the multiplier (e.g. during warmup).
        """
        if self.staircase_variable == "none":
            return 0.0, streak

        # positive streak = correct streak, negative = error streak
        if correct:
            new_rl = max(1, streak + 1) if streak >= 0 else 1
        else:
            new_rl = min(-1, streak - 1) if streak <= 0 else -1
        n = abs(new_rl)

        # step scale based on which variable we're adjusting
        if self.staircase_variable == "tower_duration":
            delta_down = settings.staircase_delta_down_ms
            delta_up = settings.staircase_delta_up_ms
            delta_max = settings.staircase_delta_max_ms
        else:
            delta_down = settings.staircase_delta_down
            delta_up = settings.staircase_delta_up
            delta_max = settings.staircase_delta_max

        r = settings.staircase_r
        base = (delta_up if correct else delta_down) * (r ** (n - 1))

        # Main-phase onset multiplier: large steps at start, decays to 1
        if 0 < mult_trial <= settings.onset_boost_trials:
            mult = (settings.staircase_M
                    * np.exp(-mult_trial / settings.staircase_tau) + 1.0)
        else:
            mult = 1.0

        return min(base * mult, delta_max), new_rl, mult


STAGES: dict[int, StageConfig] = {
    0: StageConfig(stage=0, name="BackForth", rwd_density=0.0,
                   no_rwd_density=0.0, trial_is_cued=False,
                   give_free_reward=True, both_sides_rewarded=True,
                   staircase_variable="none", staircase_start=0.0,
                   staircase_target=0.0, staircase_harder_direction="up",
                   color="blueviolet", advance_threshold=0.0),
    1: StageConfig(stage=1, name="OneSide", rwd_density=8.4,
                   no_rwd_density=0.0, trial_is_cued=False,
                   give_free_reward=True, both_sides_rewarded=False,
                   staircase_variable="none", staircase_start=0.0,
                   staircase_target=0.0, staircase_harder_direction="up",
                   color="lawngreen", advance_threshold=0.80),
    2: StageConfig(stage=2, name="+mu_nr", rwd_density=8.4, no_rwd_density=0.0,
                   trial_is_cued=False, give_free_reward=True,
                   both_sides_rewarded=False,
                   staircase_variable="minority_density", staircase_start=0.0,
                   staircase_target=1.6, staircase_harder_direction="up",
                   color="sandybrown", advance_threshold=0.70, has_warmup=True),
    3: StageConfig(stage=3, name="-LED_ms", rwd_density=8.0,
                   no_rwd_density=1.6, trial_is_cued=False,
                   give_free_reward=True, both_sides_rewarded=False,
                   staircase_variable="tower_duration", staircase_start=5000.0,
                   staircase_target=200.0, staircase_harder_direction="down",
                   color="royalblue", advance_threshold=0.70, timed_leds=True,
                   has_warmup=True),
    4: StageConfig(stage=4, name="+mu_nr_short", rwd_density=7.7,
                   no_rwd_density=1.6, trial_is_cued=False,
                   give_free_reward=True, both_sides_rewarded=False,
                   staircase_variable="minority_density", staircase_start=1.6,
                   staircase_target=2.3, staircase_harder_direction="up",
                   color="tomato", advance_threshold=0.70, timed_leds=True,
                   has_warmup=True),
    5: StageConfig(stage=5, name="Final", rwd_density=7.7,
                   no_rwd_density=2.3, trial_is_cued=False,
                   give_free_reward=True, both_sides_rewarded=False,
                   staircase_variable="none", staircase_start=0.0,
                   staircase_target=0.0, staircase_harder_direction="up",
                   color="gold", advance_threshold=0.0, timed_leds=True,
                   has_warmup=True),
}

MIN_STAGE = 0
MAX_STAGE = 5

CHECKPOINT_COLORS = ["darkviolet", "darkgreen", "midnightblue", "goldenrod",
                     "sienna"]

PHASE_BGR: dict[str, tuple[int, int, int]] = {
    "warmup": (0, 220, 220),  # gold
    "main":    (40, 160, 40),  # green
}

# Columns that must be present in a session df for plots to be plotted.
REQUIRED_COLS: frozenset[str] = frozenset({"trial", "trial_correct",
                                           "stage", "phase", "mu_nr",
                                           "led_ms", "streak"})
