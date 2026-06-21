from dataclasses import dataclass


@dataclass(frozen=True)
class StagePolicy:
    """Optional reward-modulation mechanisms."""
    jackpot: bool = False  # occasional large reward
    effort: bool = False   # difficulty-scaled reward


@dataclass
class Difficulty:
    """Active difficulty parameters for one trial."""
    mu_r: float = 0.0  # rewarded tower density (m^-1)
    mu_nr: float = 0.0  # non-rewarded tower density (m^-1)
    led_ms: int = 5000  # LED on-duration (ms)
    light_intensity: int = 255  # S1 visual cue PWM (ports 1/3), fades 255->0


@dataclass(frozen=True)
class Staircase:
    """Encapsulates one Kaernbach adaptive staircase.

    variable: column driven ("none" | "minority_density" | "tower_duration"
              | "light_intensity")
    start: initial value when stage is entered
    target: convergence target (e.g. mu_nr >= 1.6)
    harder_direction: "up" (mu_nr increases) | "down" (led_ms decreases)
    enabled: False then no steps applied like in a normal protocol
    target_acc: per-stage p* (staircase equilibrium accuracy).
    """
    variable: str = "none"
    start: float = 0.0
    target: float = 0.0
    harder_direction: str = "up"
    enabled: bool = True
    target_acc: float = 0.75

    def compute_step(self, correct: bool, streak: int,
                     boost_mult: float, settings) -> tuple[float, int, float]:
        """Compute step and updated streak for one trial.

        boost_mult: precomputed onset multiplier from OnsetBoost.
        Returns (delta, new_streak, boost_mult).
        """
        if not self.enabled or self.variable == "none":
            return 0.0, streak, 1.0

        # positive streak = correct streak, negative = error streak
        if correct:
            new_rl = max(1, streak + 1) if streak >= 0 else 1
        else:
            new_rl = min(-1, streak - 1) if streak <= 0 else -1
        n = abs(new_rl)

        # p* = staircase operating accuracy (per-stage)
        p = self.target_acc
        ratio = p / (1.0 - p)  # delta_down = delta_up * ratio
        if self.variable == "tower_duration":
            delta_up = settings.staircase_delta_up_ms
            delta_max = settings.staircase_delta_max_ms
        elif self.variable == "light_intensity":
            delta_up = settings.staircase_delta_up_intensity
            delta_max = settings.staircase_delta_max_intensity
        else:
            delta_up = settings.staircase_delta_up
            delta_max = settings.staircase_delta_max
        delta_down = delta_up * ratio

        r = settings.staircase_r
        base = (delta_up if correct else delta_down) * (r ** (n - 1))
        return min(base * boost_mult, delta_max), new_rl, boost_mult


@dataclass(frozen=True)
class StageConfig:
    stage: int
    name: str
    rwd_density: float
    no_rwd_density: float  # starting value; staircase overrides from S2
    trial_is_cued: bool  # always False in v2
    give_free_reward: bool  # always True in v2
    both_sides_rewarded: bool  # True only in Stage 0
    staircases: tuple = ()  # tuple[Staircase1, Staircase2, ...]
    color: str = "w"
    advance_threshold: float = 0.70
    advance_label: str = ""
    timed_leds: bool = False
    has_warmup: bool = False
    warmup_min_trials: int | None = None
    warmup_acc_threshold: float | None = None
    warmup_bias_threshold: float | None = None
    rescue_threshold: float | None = None
    policy: StagePolicy = StagePolicy()

    @property
    def staircase(self) -> Staircase:
        """First (and only?) staircase, or a no-op Staircase()."""
        return self.staircases[0] if self.staircases else Staircase()


STAGES: dict[int, StageConfig] = {
    0: StageConfig(
        stage=0, name="BackForth", rwd_density=0.0,
        no_rwd_density=0.0, trial_is_cued=True,
        give_free_reward=True, both_sides_rewarded=True,
        staircases=(),
        color="blueviolet", advance_threshold=0.0),
    1: StageConfig(
        stage=1, name="OneSide", rwd_density=8.4,
        no_rwd_density=0.0, trial_is_cued=True,
        give_free_reward=True, both_sides_rewarded=False,
        staircases=(Staircase(variable="light_intensity",
                              start=255, target=0,
                              harder_direction="down", target_acc=0.80),),
        color="lawngreen", advance_threshold=0.80,
        policy=StagePolicy(jackpot=True)),
    2: StageConfig(
        stage=2, name="+mu_nr", rwd_density=8.4, no_rwd_density=0.0,
        trial_is_cued=False, give_free_reward=True,
        both_sides_rewarded=False,
        staircases=(Staircase(variable="minority_density",
                              start=0.0, target=1.6,
                              harder_direction="up", target_acc=0.75),),
        color="sandybrown", advance_threshold=0.75, has_warmup=True,
        warmup_min_trials=20, warmup_acc_threshold=0.80,
        warmup_bias_threshold=0.10, rescue_threshold=0.65,
        policy=StagePolicy(jackpot=True)),
    3: StageConfig(
        stage=3, name="-LED_ms", rwd_density=8.0,
        no_rwd_density=1.6, trial_is_cued=False,
        give_free_reward=True, both_sides_rewarded=False,
        staircases=(Staircase(variable="tower_duration",
                              start=5000.0, target=200.0,
                              harder_direction="down", target_acc=0.70),),
        color="royalblue", advance_threshold=0.70, timed_leds=True,
        has_warmup=True, warmup_min_trials=10, warmup_acc_threshold=0.85,
        warmup_bias_threshold=0.10, rescue_threshold=0.60,
        policy=StagePolicy(jackpot=True)),
    4: StageConfig(
        stage=4, name="+mu_nr_short", rwd_density=7.7,
        no_rwd_density=1.6, trial_is_cued=False,
        give_free_reward=True, both_sides_rewarded=False,
        staircases=(Staircase(variable="minority_density",
                              start=1.6, target=2.3,
                              harder_direction="up", target_acc=0.70),),
        color="tomato", advance_threshold=0.70, timed_leds=True,
        has_warmup=True, warmup_min_trials=10, warmup_acc_threshold=0.85,
        warmup_bias_threshold=0.10, rescue_threshold=0.60,
        policy=StagePolicy(jackpot=True)),
    5: StageConfig(
        stage=5, name="Final", rwd_density=7.7,
        no_rwd_density=2.3, trial_is_cued=False,
        give_free_reward=True, both_sides_rewarded=False,
        staircases=(),
        color="gold", advance_threshold=0.0, timed_leds=True,
        has_warmup=True, warmup_min_trials=10, warmup_acc_threshold=0.85,
        warmup_bias_threshold=0.10, rescue_threshold=0.55),
}

MIN_STAGE = 0
MAX_STAGE = 5

CHECKPOINT_COLORS = ["darkviolet", "darkgreen", "midnightblue", "goldenrod",
                     "sienna"]

PHASE_BGR: dict[str, tuple[int, int, int]] = {
    "warmup": (0, 220, 220),
    "main":    (40, 160, 40),
}

# Columns that must be present in a session df for plots to be plotted.
REQUIRED_COLS: frozenset[str] = frozenset({"trial", "trial_correct",
                                           "stage", "phase", "mu_nr",
                                           "led_ms", "streak"})
