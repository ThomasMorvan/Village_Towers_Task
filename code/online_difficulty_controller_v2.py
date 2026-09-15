"""Difficulty controller for the V2 (speed-gated) Towers task."""

from collections import deque

from online_difficulty_controller import AdaptationEvent, OnsetBoost, Warmup
from task_stages import Difficulty
from task_stages_v2 import MAX_STAGE_V2, MIN_STAGE_V2, STAGES_V2


class OnlineDifficultyControllerV2:
    """Within-session difficulty controller for TowersTaskV2."""

    STAGE_SETTING = "v2_stage"

    def __init__(self) -> None:
        self.stage: int = MIN_STAGE_V2
        self.checkpoint: int = 0
        self.checkpoint_floor: float = 0.0
        self.difficulty: Difficulty = Difficulty()
        self.phase: str = "main"
        self.last_delta: float = 0.0
        self.last_boost: float = 1.0
        self._streak: int = 0
        self._perf_window: deque = deque()
        self._carryover_n: int = 0
        self._n_at_target: int = 0
        self._warmup: Warmup | None = None
        self._boost: OnsetBoost | None = None

    @property
    def config(self):
        return STAGES_V2[self.stage]

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def rescue_active(self) -> bool:
        return False

    @property
    def warmup_n(self) -> int:
        return self._warmup.n if self._warmup else 0

    @property
    def warmup_min(self) -> int:
        return self._warmup.min_trials if self._warmup else 0

    @property
    def warmup_acc(self) -> float:
        return self._warmup.acc if self._warmup else 0.0

    @property
    def warmup_bias(self) -> float:
        return self._warmup.bias if self._warmup else 0.0

    @property
    def rolling_acc(self) -> float | None:
        if not self._perf_window:
            return None
        return sum(self._perf_window) / len(self._perf_window)

    @property
    def max_speed(self) -> float:
        return self.difficulty.max_speed_cm_s

    @property
    def gate_active(self) -> bool:
        """Whether the speed contingency applies to this trial."""
        return self.phase == "main"

    def start(self, settings) -> None:
        """Restore stage and threshold, then open a warmup if the stage has
        one. Reads `v2_stage` / `last_max_speed`, never V1's `stage`."""
        self.stage = min(max(int(getattr(settings, self.STAGE_SETTING, 0)),
                             MIN_STAGE_V2), MAX_STAGE_V2)
        cfg = self.config
        sc = cfg.staircase

        self.difficulty = Difficulty(
            mu_r=cfg.rwd_density,
            mu_nr=cfg.no_rwd_density,
            led_ms=int(getattr(settings, "min_tower_duration", 100)),
            end_dead_zone_cm=float(getattr(settings, "last_dead_zone_cm",
                                           0.0)))

        if sc.variable == "max_speed":
            last = float(getattr(settings, "last_max_speed", sc.start))
            self.difficulty.max_speed_cm_s = min(max(last, sc.target),
                                                 sc.start)
        else:
            self.difficulty.max_speed_cm_s = STAGES_V2[0].staircase.target

        self._perf_window = deque(
            list(getattr(settings, "last_perf_window", []) or []),
            maxlen=int(getattr(settings, "acc_window", 40)))
        self._carryover_n = len(self._perf_window)
        self._n_at_target = 0
        self._streak = 0
        self._boost = OnsetBoost(
            M=float(getattr(settings, "staircase_M", 4.0)),
            tau=float(getattr(settings, "staircase_tau", 10.0)),
            n_trials=int(getattr(settings, "onset_boost_trials", 30)),
            enabled=True)
        self._reset_warmup(settings)

    def _reset_warmup(self, settings) -> None:
        cfg = self.config
        if not cfg.has_warmup:
            self.phase = "main"
            self._warmup = None
            return
        self.phase = "warmup"
        self._warmup = Warmup(
            min_trials=int(cfg.warmup_min_trials or 10),
            acc_threshold=float(cfg.warmup_acc_threshold or 0.85),
            bias_threshold=float(cfg.warmup_bias_threshold or 0.10),
            bias_window=int(getattr(settings, "warmup_bias_window", 20)))

    def after_trial(self, correct: bool, side, settings,
                    bias: float = 0.0) -> AdaptationEvent:
        cfg = self.config

        if self.phase == "warmup" and self._warmup is not None:
            self._warmup.record(side, correct)
            if self._warmup.passed:
                self.phase = "main"
                if self._boost:
                    self._boost.reset()
                return AdaptationEvent(warmup_passed=True)
            return AdaptationEvent()

        self._perf_window.append(int(correct))

        sc = cfg.staircase
        if sc.variable == "max_speed":
            boost = self._boost.next() if self._boost else 1.0
            delta, self._streak, self.last_boost = sc.compute_step(
                correct, self._streak, boost, settings)
            self.last_delta = delta
            if correct:
                self.difficulty.max_speed_cm_s = max(
                    self.difficulty.max_speed_cm_s - delta, sc.target)
            else:
                self.difficulty.max_speed_cm_s = min(
                    self.difficulty.max_speed_cm_s + delta, sc.start)

        return self._check_graduation(settings)

    def _check_graduation(self, settings) -> AdaptationEvent:
        """Generic rule: staircase target reached (within tolerance) while
        rolling accuracy holds."""
        cfg = self.config
        if self.stage >= MAX_STAGE_V2:
            return AdaptationEvent()
        acc = self.rolling_acc
        if acc is None or len(self._perf_window) < self._perf_window.maxlen:
            return AdaptationEvent()
        sc = cfg.staircase
        if sc.variable != "max_speed":
            return AdaptationEvent()
        tol = sc.grad_tol(settings)

        if self.difficulty.max_speed_cm_s <= sc.target + sc.grad_tol(
                settings, size=2.0):
            self._n_at_target += 1
        else:
            self._n_at_target = 0
        if (acc >= cfg.advance_threshold
                and self._n_at_target >= self._perf_window.maxlen
                and self.difficulty.max_speed_cm_s <= sc.target + tol):
            self.stage += 1
            self.checkpoint += 1
            self._streak = 0
            self._n_at_target = 0
            self._perf_window.clear()
            self._reset_warmup(settings)
            self.difficulty.max_speed_cm_s = STAGES_V2[0].staircase.target
            return AdaptationEvent(stage_advanced_to=self.stage)
        return AdaptationEvent()
