"""Difficulty controller for the V2 (speed-gated) Towers task."""

from collections import deque

from online_difficulty_controller import AdaptationEvent, Warmup
from task_stages import STAGES as STAGES_V1, Difficulty
from task_stages_v2 import (MAX_STAGE_V2, MIN_STAGE_V2, SLOW_FRAC,
                            STAGE_GATE, STAGES_V2)


class OnlineDifficultyControllerV2:
    """Within-session controller for TowersTaskV2_1."""

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
        self._slow_window: deque = deque()
        self._carryover_n: int = 0
        self.run_was_slow: bool | None = None
        self._warmup: Warmup | None = None

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
    def slow_frac(self) -> float | None:
        if not self._slow_window:
            return None
        return sum(self._slow_window) / len(self._slow_window)

    @property
    def max_speed(self) -> float:
        return self.difficulty.max_speed_cm_s

    @property
    def gate_active(self) -> bool:
        return self.phase == "main"

    def start(self, settings) -> None:
        self.stage = min(max(int(getattr(settings, self.STAGE_SETTING, 0)),
                             MIN_STAGE_V2), MAX_STAGE_V2)
        cfg = self.config
        self.difficulty = Difficulty(
            mu_r=cfg.rwd_density, mu_nr=cfg.no_rwd_density,
            led_ms=int(getattr(settings, "min_tower_duration", 100)),
            end_dead_zone_cm=float(STAGES_V1[2].staircase.target),
            max_speed_cm_s=STAGE_GATE[self.stage])
        win = int(getattr(settings, "acc_window", 40))
        self._perf_window = deque(
            list(getattr(settings, "last_perf_window", []) or []), maxlen=win)
        self._slow_window = deque(
            list(getattr(settings, "last_slow_window", []) or []), maxlen=win)
        self._carryover_n = len(self._perf_window)
        self._streak = 0
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
        if self.phase == "warmup" and self._warmup is not None:
            self._warmup.record(side, correct)
            if self._warmup.passed():
                self.phase = "main"
                return AdaptationEvent(warmup_passed=True)
            return AdaptationEvent()

        self._perf_window.append(int(correct))
        if self.run_was_slow is not None:
            self._slow_window.append(int(self.run_was_slow))
        self._streak = (max(1, self._streak + 1) if correct
                        else min(-1, self._streak - 1))
        return self._check_graduation(settings)

    def _check_graduation(self, settings) -> AdaptationEvent:
        cfg = self.config
        if self.stage >= MAX_STAGE_V2:
            return AdaptationEvent()
        acc, slow = self.rolling_acc, self.slow_frac
        if (acc is None or slow is None
                or len(self._slow_window) < self._slow_window.maxlen):
            return AdaptationEvent()
        if slow >= SLOW_FRAC and acc >= cfg.advance_threshold:
            self.stage += 1
            self.checkpoint += 1
            self._streak = 0
            self._perf_window.clear()
            self._slow_window.clear()
            self.difficulty.max_speed_cm_s = STAGE_GATE[self.stage]
            self._reset_warmup(settings)
            return AdaptationEvent(stage_advanced_to=self.stage)
        return AdaptationEvent()
