"""TowersTask V2: cues are suppressed while the animal runs too fast."""

import time

from online_difficulty_controller_v2 import OnlineDifficultyControllerV2
from online_speed_estimator import OnlineSpeedEstimator, px_to_cm_fit
from task_stages_v2 import STAGES_V2
from tower_task import TowersTask
from village.scripts.log import log


class TowersTaskV2(TowersTask):
    """TowersTask with a speed-gated cue contingency."""

    TH_LO_DIFF = 10.0

    def __init__(self):
        super().__init__()
        self._odc = OnlineDifficultyControllerV2()
        self.speed_estimator = OnlineSpeedEstimator()
        self._to_cm = None
        self._suppress_cues = False
        self._led_suppressed_log: list = []
        self._base_adv: list = []
        self._reset_speed_stats()

    def _reset_speed_stats(self) -> None:
        self._n_frames = 0
        self._n_fast = 0
        self._n_suppressed = 0
        self._max_speed = 0.0

    def start(self):
        self.settings.proximity_trigger = False
        super().start()
        try:
            self._to_cm = px_to_cm_fit(self.led_positions)
        except Exception:
            self._to_cm = None
            log.error("[v2] px->cm failed; speed gate disabled",
                      exception=True)
        self._apply_speed_threshold()
        self._reset_speed_stats()
        log.info(f"[v2] stage={self._odc.stage} "
                 f"({STAGES_V2[self._odc.stage].name}) "
                 f"max_speed={self._odc.max_speed:.1f} cm/s")

    @property
    def stage_cfg(self):
        return STAGES_V2[self._odc.stage]

    def _apply_stage(self, stage: int) -> None:
        """V2 equivalent of TowersTask._apply_stage.
        """
        cfg = STAGES_V2[stage]
        self.trial_is_cued = cfg.trial_is_cued
        self.give_free_reward = cfg.give_free_reward

        diff = self._odc.difficulty
        self.led_picker.update_mu(diff.mu_r, diff.mu_nr)
        self.led_picker.update_dead_zone(diff.end_dead_zone_cm)

        self.settings.v2_stage = stage
        self.settings.last_max_speed = float(diff.max_speed_cm_s)

    def _apply_speed_threshold(self) -> None:
        """Push the staircase's current threshold into the estimator."""
        hi = float(self._odc.max_speed)
        self.speed_estimator.th_hi = hi
        self.speed_estimator.th_lo = hi - self.TH_LO_DIFF
        self.speed_estimator.reset()

    def softcode_callback(self):
        if self._to_cm is not None:
            x = self.current_x
            pos_cm = None if x is None or x < 0 else self._to_cm(x)
            t = getattr(self.cam_box, "camera_timestamp", None) or time.time()
            try:
                fast = self.speed_estimator.update(t, pos_cm)
                self._suppress_cues = fast and self._odc.gate_active
            except Exception:
                log.error("[v2] estimator failed; gate disabled",
                          exception=True)
                self._to_cm = None
                self._suppress_cues = False
            self._n_frames += 1
            self._n_fast += int(self._suppress_cues)
            speed = self.speed_estimator.speed
            if speed is not None:
                self._max_speed = max(self._max_speed, speed)
        n_before = len(self._led_on_log)
        super().softcode_callback()
        if self._suppress_cues and len(self._led_on_log) > n_before:
            self._led_suppressed_log.extend(self._led_on_log[n_before:])
            del self._led_on_log[n_before:]
        self._draw_speed()

    def execute_function(self, n: int):
        """Suppress cue-lighting softcodes while the gate says FAST.

        Intercepting here rather than reimplementing the parent's trigger
        logic keeps the proximity bookkeeping (`_furthest_x`, trigger list,
        used/available sets) identical to V1. The trigger is consumed but the
        light does not come on.
        """
        if self._suppress_cues and n in (self.SOFTCODE_SINGLE_LED_ON,
                                         self.SOFTCODE_ALL_LEDS_ON):
            self._n_suppressed += 1
            return None
        return super().execute_function(n)

    def _update_hud(self) -> None:
        stage = self._odc.stage
        cfg = STAGES_V2[stage]
        acc = self._odc.rolling_acc
        acc_txt = f"{acc * 100:.0f}" if acc is not None else "?"
        sc = cfg.staircase
        if sc.variable == "max_speed":
            adv = [("Acc:", f" {acc_txt}/{cfg.advance_threshold * 100:.0f}%",
                    acc is not None and acc >= cfg.advance_threshold),
                   ("MaxSpd:", f" {self._odc.max_speed:.1f}/{sc.target:.0f}",
                    self._odc.max_speed <= sc.target + sc.grad_tol(
                        self.settings))]
        else:
            adv = [("", "  V2 final", True)]

        self.cam_box.items_to_draw["hud"] = {
            "phase": self._odc.phase,
            "stage": stage,
            "stage_name": cfg.name,
            "difficulty": self._odc.difficulty,
            "checkpoint": self._odc.checkpoint,
            "checkpoint_floor": self._odc.checkpoint_floor,
            "streak": self._odc.streak,
            "rolling_acc": acc,
            "warmup_trial": (self._odc.warmup_n, self._odc.warmup_min),
            "adv_label": adv,
            "perf_window": list(self._odc._perf_window),
            "carryover_n": self._odc._carryover_n,
        }
        self._base_adv = list(adv)

    def _draw_speed(self) -> None:
        hud = self.cam_box.items_to_draw.get("hud")
        if hud is None:
            return
        speed = self.speed_estimator.speed
        value = "   --" if speed is None else f" {speed:5.1f}"
        hud["adv_label"] = self._base_adv + [
            ("Speed:", f"{value} cm/s", not self._suppress_cues)]

    def after_trial(self):
        n_shown = sum(len(e[1]) for e in self._led_on_log)
        n_hidden = sum(len(e[1]) for e in self._led_suppressed_log)
        super().after_trial()
        n = max(self._n_frames, 1)
        self.register_value("v2_stage", self._odc.stage)
        self.register_value("max_speed_cm_s",
                            round(self._odc.max_speed, 2))
        self.register_value("cues_suppressed", self._n_suppressed)
        self.register_value("frac_fast", round(self._n_fast / n, 3))
        self.register_value("trial_max_speed", round(self._max_speed, 1))
        # What the animal was actually shown, as opposed to what was drawn.
        self.register_value("leds_delivered", n_shown)
        self.register_value("leds_suppressed", n_hidden)
        self.register_value("led_suppressed_times", self._led_suppressed_log)
        log.info(f"[v2] max_speed={self._odc.max_speed:.1f} "
                 f"suppressed={self._n_suppressed} "
                 f"fast={self._n_fast / n:.2f} "
                 f"peak={self._max_speed:.1f}cm/s")
        self._led_suppressed_log = []
        self._reset_speed_stats()
        self._apply_speed_threshold()
