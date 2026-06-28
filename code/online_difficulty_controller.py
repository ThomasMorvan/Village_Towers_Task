from collections import deque
from dataclasses import dataclass
import numpy as np

from task_stages import STAGES, MAX_STAGE, Difficulty


@dataclass
class AdaptationEvent:
    """Small event returned by after_trial() to tell Task what to do
    (advance stage, start warmup, trigger rescue, etc.)."""
    warmup_passed: bool = False
    stage_advanced_to: int | None = None
    rescue_triggered: bool = False
    rescue_ended: bool = False


class Warmup:
    """Warmup phase tracker at start of each session
    (easy trials to warm up)."""
    def __init__(self, min_trials: int, acc_threshold: float,
                 bias_threshold: float, enabled: bool = True) -> None:
        self.min_trials = min_trials
        self.acc_threshold = acc_threshold
        self.bias_threshold = bias_threshold
        self.enabled = enabled
        self._perf: deque = deque()

    def reset(self, maxlen: int | None = None) -> None:
        self._perf = deque(maxlen=maxlen)

    def record(self, side, correct: bool) -> None:
        self._perf.append((side, correct))

    @property
    def n(self) -> int:
        return len(self._perf)

    @property
    def acc(self) -> float:
        n = len(self._perf)
        return sum(c for _, c in self._perf) / n if n else 0.0

    @property
    def bias(self) -> float:
        from left_or_right import TrialSide
        left_c = [c for s, c in self._perf if s == TrialSide.LEFT]
        right_c = [c for s, c in self._perf if s == TrialSide.RIGHT]
        return (abs(sum(left_c) / len(left_c) - sum(right_c) / len(right_c))
                if left_c and right_c else 1.0)

    def passed(self) -> bool:
        if not self.enabled:
            return True
        # Compare at percent resolution so the gate is easier and match HUD
        return (self.n >= self.min_trials
                and round(self.acc * 100) >= round(self.acc_threshold * 100)
                and round(self.bias * 100) <= round(self.bias_threshold * 100))


class OnsetBoost:
    """Exponential onset multiplier for staircase steps at main-phase start.
    First trials steps are boosted to speed up convergence
    then decays back to normal over time.

    next() returns M·exp(-t/tau)+1 for the first n_trials calls, then 1.0.
    enabled=False --> next() always returns 1.0 (no boost).
    """

    def __init__(self, M: float, tau: float, n_trials: int,
                 enabled: bool = True) -> None:
        self.M = M
        self.tau = tau
        self.n_trials = n_trials
        self.enabled = enabled
        self._t: int = 0

    def reset(self) -> None:
        self._t = 0

    def next(self) -> float:
        if not self.enabled:
            return 1.0
        self._t += 1
        if self._t <= self.n_trials:
            return self.M * np.exp(-self._t / self.tau) + 1.0
        return 1.0


class OnlineDifficultyController:
    """Within-session trial-by-trial difficulty controller.

    Owns all staircase state and intra-session checkpoint logic.
    Task reads public attributes (stage, difficulty, phase, …) and
    receives an AdaptationEvent from after_trial() describing what happened,
    so it can update device objects (LedPicker, HUD)."""

    def __init__(self) -> None:
        self.stage: int = 0
        self.checkpoint: int = 0
        self.checkpoint_floor: float = 0.0
        self.difficulty: Difficulty = Difficulty()
        self.phase: str = "main"
        self.last_delta: float = 0.0
        self.last_boost: float = 1.0
        self._streak: int = 0
        self._perf_window: deque = deque()
        self._rescue_trials_left: int = 0

        self._warmup: Warmup | None = None
        self._boost: OnsetBoost | None = None

    @property
    def config(self):
        return STAGES[self.stage]

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def rescue_active(self) -> bool:
        return self._rescue_trials_left > 0

    @property
    def warmup_n(self) -> int:
        return self._warmup.n if self._warmup else 0

    @property
    def warmup_min(self) -> int:
        """Per-stage warmup trial target (0 if no warmup)."""
        return self._warmup.min_trials if self._warmup else 0

    @property
    def warmup_acc(self) -> float:
        return self._warmup.acc if self._warmup else 0.0

    @property
    def warmup_bias(self) -> float:
        return self._warmup.bias if self._warmup else 1.0

    @property
    def rolling_acc(self) -> float | None:
        return (sum(self._perf_window) / len(self._perf_window)
                if self._perf_window else None)

    def start(self, settings) -> None:
        """Restore from settings; initialise difficulty with resume logic."""
        self.stage = min(int(getattr(settings, "stage", 0)), MAX_STAGE)
        self.checkpoint = int(getattr(settings, "checkpoint", 0))
        self.checkpoint_floor = float(getattr(settings, "checkpoint_floor",
                                              0.0))

        floor = self.checkpoint_floor
        resume = bool(getattr(settings, "resume_from_last", True))
        mu_r = STAGES[self.stage].rwd_density
        min_ms = int(getattr(settings, "min_tower_duration", 200))
        if self.stage in (2, 4):
            last = (float(getattr(settings, "last_mu_nr", floor)) if resume
                    else floor)
            led_ms = 5000 if self.stage == 2 else min_ms
            self.difficulty = Difficulty(mu_r=mu_r, mu_nr=max(last, floor),
                                         led_ms=led_ms)
        elif self.stage == 3:
            last_ms = (int(getattr(settings, "last_led_ms", 5000)) if resume
                       else 5000)
            self.difficulty = Difficulty(mu_r=mu_r,
                                         mu_nr=STAGES[3].no_rwd_density,
                                         led_ms=last_ms if last_ms else 5000)
        elif self.stage == 5:
            self.difficulty = Difficulty(mu_r=mu_r,
                                         mu_nr=STAGES[5].no_rwd_density,
                                         led_ms=min_ms)
        else:
            self.difficulty = Difficulty(mu_r=mu_r)

        # Visual cue intensity (S1 staircase).
        li_max = int(getattr(settings, "light_intensity_high", 255))
        if self.stage == 1 and resume:
            self.difficulty.light_intensity = int(
                getattr(settings, "last_light_intensity", li_max))
        else:
            self.difficulty.light_intensity = li_max

        self._streak = 0
        self._perf_window = deque(maxlen=int(settings.acc_window))
        if resume:  # carry rolling-acc window across sessions
            self._perf_window.extend(int(c)
                                     for c in getattr(settings,
                                                      "last_perf_window",
                                                      []) or [])
        self._rescue_trials_left = 0
        self.last_delta = 0.0
        self.last_boost = 1.0

        cfg = STAGES[self.stage]
        self.phase = "warmup" if cfg.has_warmup else "main"
        self._reset_warmup()
        self._reset_boost(settings)

    def after_trial(self, correct: bool, side, settings,
                    bias: float = 0.0) -> AdaptationEvent:
        """Process one trial outcome; return Event for Task to act on.

        bias: abs(empirical_p_right - 0.5), used for S1 advance check.
              Caller (Task) computes this from left_or_right.current_empR.
        """
        self.last_delta = 0.0
        self.last_boost = 1.0

        if self.phase == "warmup":
            return self._check_warmup(side, correct)

        # Feed main-phase trials to the rolling window.
        self._perf_window.append(int(correct))
        cfg = self.config

        if cfg.rescue_threshold is not None:
            in_block = self._rescue_trials_left > 0
            rescue_event = self._check_rescue(settings, cfg.rescue_threshold)
            if in_block or rescue_event.rescue_triggered:
                return rescue_event

        if cfg.staircase.variable == "none":
            if self.stage == 1:
                return self._check_s1_advance(settings, bias)
            return AdaptationEvent()

        boost_mult = self._boost.next() if self._boost else 1.0
        self.last_boost = boost_mult
        delta, self._streak, _ = cfg.staircase.compute_step(
            correct, self._streak, boost_mult, settings)
        self.last_delta = delta
        self._apply_staircase_delta(delta, correct, settings)
        if self.stage == 1:
            return self._check_s1_advance(settings, bias)
        return self._check_checkpoint(settings)

    def _reset_warmup(self) -> None:
        cfg = self.config
        min_trials = int(cfg.warmup_min_trials
                         if cfg.warmup_min_trials is not None else 10)
        acc = float(cfg.warmup_acc_threshold
                    if cfg.warmup_acc_threshold is not None else 0.85)
        bias = float(cfg.warmup_bias_threshold
                     if cfg.warmup_bias_threshold is not None else 0.10)
        self._warmup = Warmup(min_trials=min_trials,
                              acc_threshold=acc,
                              bias_threshold=bias,
                              enabled=cfg.has_warmup)
        self._warmup.reset()

    def _reset_boost(self, settings) -> None:
        self._boost = OnsetBoost(M=float(settings.staircase_M),
                                 tau=float(settings.staircase_tau),
                                 n_trials=int(settings.onset_boost_trials))

    def _pass_checkpoint(self, to_stage: int, settings) -> AdaptationEvent:
        self.checkpoint = to_stage - 1
        new_start = STAGES[to_stage].staircase.start
        self.checkpoint_floor = new_start
        prev = self.difficulty
        mu_r = STAGES[to_stage].rwd_density

        if to_stage == 3:
            self.difficulty = Difficulty(mu_r=mu_r, mu_nr=prev.mu_nr,
                                         led_ms=int(new_start))
        elif to_stage in (2, 4):
            self.difficulty = Difficulty(mu_r=mu_r, mu_nr=new_start,
                                         led_ms=prev.led_ms)
        else:
            self.difficulty = Difficulty(mu_r=mu_r, mu_nr=prev.mu_nr,
                                         led_ms=prev.led_ms)

        self._streak = 0
        self._perf_window = deque(maxlen=int(settings.acc_window))
        self._rescue_trials_left = 0
        self.stage = to_stage
        self.phase = "main"
        if settings.onset_boost_on_graduation:
            self._reset_boost(settings)

        print(f"   * [ODC] Checkpoint {self.checkpoint} passed!"
              f" -> Stage {to_stage} (floor={self.checkpoint_floor:.3f})")
        return AdaptationEvent(stage_advanced_to=to_stage)

    def _check_warmup(self, side, correct) -> AdaptationEvent:
        self._warmup.record(side, correct)
        if self._warmup.passed():
            self.phase = "main"
            if self._boost:
                self._boost.reset()
            print(f"   * [ODC] Warmup passed! n={self._warmup.n}, "
                  f"acc={self._warmup.acc:.0%}, bias={self._warmup.bias:.0%}")
            return AdaptationEvent(warmup_passed=True)
        return AdaptationEvent()

    def _check_rescue(self, settings, threshold: float) -> AdaptationEvent:
        if self._rescue_trials_left > 0:
            self._rescue_trials_left -= 1
            if self._rescue_trials_left == 0:
                print("   * [ODC] Rescue complete, returning to main task")
                return AdaptationEvent(rescue_ended=True)
            return AdaptationEvent()
        if (getattr(settings, "rescue_enabled", False)
                and len(self._perf_window) >= settings.acc_window
                and (sum(self._perf_window) / len(self._perf_window))
                < threshold):
            self._rescue_trials_left = int(settings.rescue_block_size)
            print(f"   * [ODC] Rescue triggered!"
                  f" {settings.rescue_block_size} easy trials")
            return AdaptationEvent(rescue_triggered=True)
        return AdaptationEvent()

    def _check_s1_advance(self, settings, bias: float) -> AdaptationEvent:
        if (len(self._perf_window) >= settings.acc_window
                and self.stage == 1
                and getattr(settings, "stage", 0) <= MAX_STAGE):
            rolling_acc = sum(self._perf_window) / len(self._perf_window)
            tol = self.config.staircase.grad_tol(settings)
            if (rolling_acc >= self.config.advance_threshold and bias <= 0.10
                    and self.difficulty.light_intensity
                    <= self.config.staircase.target + tol):
                return self._pass_checkpoint(to_stage=2, settings=settings)
        return AdaptationEvent()

    def _apply_staircase_delta(self, delta: float, correct: bool,
                               settings) -> None:
        cfg = self.config
        var = cfg.staircase.variable
        if var == "minority_density":
            if correct:
                self.difficulty.mu_nr = min(
                    self.difficulty.mu_nr + delta, cfg.staircase.target)
            else:
                self.difficulty.mu_nr = max(
                    self.difficulty.mu_nr - delta, self.checkpoint_floor)

        elif var == "tower_duration":
            if correct:
                self.difficulty.led_ms = max(
                    self.difficulty.led_ms - int(delta),
                    int(settings.min_tower_duration))
            else:
                self.difficulty.led_ms = min(
                    self.difficulty.led_ms + int(delta),
                    int(self.checkpoint_floor))

        elif var == "light_intensity":
            if correct:
                self.difficulty.light_intensity = max(
                    self.difficulty.light_intensity - int(delta),
                    int(cfg.staircase.target))
            else:
                self.difficulty.light_intensity = min(
                    self.difficulty.light_intensity + int(delta),
                    int(settings.light_intensity_high))

    def _check_checkpoint(self, settings) -> AdaptationEvent:
        if getattr(settings, "stage", 0) > MAX_STAGE:
            return AdaptationEvent()
        if len(self._perf_window) < settings.acc_window:
            return AdaptationEvent()

        rolling_acc = sum(self._perf_window) / len(self._perf_window)
        cfg = self.config
        tol = cfg.staircase.grad_tol(settings)

        if (self.stage == 2
                and rolling_acc >= cfg.advance_threshold
                and self.difficulty.mu_nr >= cfg.staircase.target - tol):
            return self._pass_checkpoint(to_stage=3, settings=settings)

        if (self.stage == 3
                and rolling_acc >= cfg.advance_threshold
                and self.difficulty.led_ms <= settings.min_tower_duration + tol):
            return self._pass_checkpoint(to_stage=4, settings=settings)

        if (self.stage == 4
                and rolling_acc >= cfg.advance_threshold
                and self.difficulty.mu_nr >= cfg.staircase.target - tol):
            return self._pass_checkpoint(to_stage=5, settings=settings)

        return AdaptationEvent()
