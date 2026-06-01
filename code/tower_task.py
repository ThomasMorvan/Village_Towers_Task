
from collections import deque
import numpy as np
from village.custom_classes.task import (BpodEvent as Event,
                                         BpodOutput as Output)
from village.settings import settings
from village.manager import manager
from tower_task_base import TowersTaskBase
from left_or_right import LeftOrRight, TrialSide, TrialResult
from LEDpicker import LedPicker
from task_stages import STAGES, MAX_STAGE, Difficulty


class TowersTask(TowersTaskBase):
    """Towers Task."""

    def __init__(self):
        super().__init__()
        self.led_on_duration = 0.2  # Led on in (s)  # TODO: add in settings
        self._furthest_x = 641  # FIXME: self.cam_box.width not initialized yet
        self._this_trial_leds = {TrialSide.LEFT: np.array([], dtype=int),
                                 TrialSide.RIGHT: np.array([], dtype=int)}
        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []  # sorted list of (trigger_x, led_idx)
        self.next_trigger = 641   # display only: x position of leds
        self.distance_offset = 50  # FIXME: px in front of centroid for trigger
        self.left_or_right = LeftOrRight()
        self.current_trial_rwd_side: TrialSide = TrialSide.NONE
        self.is_trial_correct: bool = None
        self.animal_trace = deque(maxlen=25*5)

        # Placeholder, recreated in start() once settings are loaded
        self.led_picker = LedPicker(rwd_density=0.0, no_rwd_density=0.0)

        self.trial_is_cued = False
        self.give_free_reward = False

        # Difficulty step params
        self._current_stage: int = 0
        self._difficulty: Difficulty = Difficulty()
        self._checkpoint: int = 0
        self._checkpoint_floor: float = 0.0
        self._streak: int = 0
        self._warmup_perf: list = []  # (TrialSide, correct) per warmup trial
        self._main_trial: int = 0
        self._perf_window: deque = deque()  # maxlen set in start()
        self._phase: str = "warmup"
        self._rescue_trials_left: int = 0
        self._last_delta: float = 0.0
        self._last_boost: float = 1.0

    def _to_strip_indices(self, leds: np.ndarray, side: TrialSide
                          ) -> np.ndarray:
        """Map LedPicker indices (0-71) to physical strip indices,
        because strip goes from [0] start-right to [71] end-right,
        then 10 empty LEDs, then [83] end-left to [154] start-left.

        Right side: 0-71 maps directly (entry end = index 0).
        Left side: reversed and shifted (index 0 maps to the entry end of the
        corridor (physical 154), index 71 maps to the far end (physical 83).
          physical = LEFT_SIDE_OFFSET + (NUM_LEDS - 1) - leds
        """
        LEFT_SIDE_OFFSET = 83
        if side == TrialSide.LEFT:
            return LEFT_SIDE_OFFSET + (self.led_picker.NUM_LEDS - 1) - leds
        return leds

    def _update_hud(self) -> None:
        """Update cam_box items to draw"""
        stage = self._current_stage
        cfg = STAGES[stage]
        rolling_acc = (sum(self._perf_window) / len(self._perf_window)
                       if self._perf_window else None)
        acc_ok = (rolling_acc is not None
                  and rolling_acc >= cfg.advance_threshold)
        acc_pct = f"{rolling_acc*100:.0f}" if rolling_acc is not None else "?"
        thr_pct = f"{cfg.advance_threshold*100:.0f}"

        if self._phase == "warmup":
            n = len(self._warmup_perf)
            min_t = int(getattr(self.settings, "warmup_min_trials", 10))
            acc_thr = float(getattr(self.settings, "warmup_acc_threshold", 0.85))
            bias_thr = float(getattr(self.settings, "warmup_bias_threshold", 0.10))
            wm_acc = (sum(c for _, c in self._warmup_perf) / n if n else 0.0)
            left_c = [c for s, c in self._warmup_perf if s == TrialSide.LEFT]
            right_c = [c for s, c in self._warmup_perf if s == TrialSide.RIGHT]
            wm_bias = (abs(sum(left_c) / len(left_c) - sum(right_c) / len(right_c))
                       if left_c and right_c else 1.0)
            adv_label = [
                ("W-trials:", f" {n}/{min_t}", n >= min_t),
                ("W-acc:", f" {wm_acc*100:.0f}/{acc_thr*100:.0f}%",
                 wm_acc >= acc_thr),
                ("W-bias:", f" {wm_bias*100:.0f}/{bias_thr*100:.0f}%",
                 wm_bias <= bias_thr),
            ]

        elif stage == 0:
            n_valid = int(getattr(self.settings, "s0_valid_sessions", 0))
            trials_lbl = ("Trials:", f" {self.current_trial}/40",
                          self.current_trial >= 40)
            n_req = int(getattr(self.settings, "s0_required_sessions", 2))
            sessions_lbl = ("Sessions:", f" {n_valid}/{n_req}",
                            n_valid >= n_req)
            adv_label = [trials_lbl, sessions_lbl]

        elif stage == 1:
            bias = abs(self.left_or_right.current_empR - 0.5)
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            bias_lbl = ("Bias:", f" {bias*100:.0f}/10%", bias*100 <= 10)
            adv_label = [acc_lbl, bias_lbl]
        elif stage == 2:
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            mu_nr_lbl = ("mu_nr:", (f" {self._difficulty.mu_nr:.2f}/"
                                    f"{cfg.staircase_target:.1f}"),
                         self._difficulty.mu_nr >= cfg.staircase_target)
            adv_label = [acc_lbl, mu_nr_lbl]
        elif stage == 3:
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            led_ms_lbl = ("LED ms:", (f" {self._difficulty.led_ms:.0f}/"
                                      f"{self.settings.min_tower_duration:.0f}"
                                      ),
                          (self._difficulty.led_ms <=
                           self.settings.min_tower_duration))
            adv_label = [acc_lbl, led_ms_lbl]
        elif stage == 4:
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            mu_nr_lbl = ("mu_nr:", (f" {self._difficulty.mu_nr:.2f}/"
                                    f"{cfg.staircase_target:.1f}"),
                         self._difficulty.mu_nr >= cfg.staircase_target)
            adv_label = [acc_lbl, mu_nr_lbl]
        elif stage == 5:
            adv_label = [("", "  Final stage", True)]
        else:
            adv_label = [("", "  Done", True)]

        self.cam_box.items_to_draw["hud"] = {
            "phase":            self._phase,
            "stage":            stage,
            "stage_name":       cfg.name,
            "difficulty":       self._difficulty,
            "checkpoint":       self._checkpoint,
            "checkpoint_floor": self._checkpoint_floor,
            "streak":           self._streak,
            "rolling_acc":      rolling_acc,
            "warmup_trial":     (len(self._warmup_perf),
                                 int(getattr(self.settings,
                                             "warmup_min_trials", 10))),
            "adv_label":        adv_label,
        }

    def _apply_stage(self, stage: int) -> None:
        """Change difficulty parameters according to stage config."""
        cfg = STAGES[stage]
        self._current_stage = stage
        self.trial_is_cued = cfg.trial_is_cued
        self.give_free_reward = cfg.give_free_reward

        if stage == 0:
            self._difficulty = Difficulty()
        elif stage == 1:
            self._difficulty = Difficulty(
                mu_r=cfg.rwd_density, mu_nr=0.0, led_ms=5000)
            self.led_picker.update_mu(cfg.rwd_density, 0.0)
        elif stage == 2:
            mu_nr = max(self._difficulty.mu_nr, self._checkpoint_floor)
            self._difficulty = Difficulty(mu_r=cfg.rwd_density,
                                          mu_nr=mu_nr,
                                          led_ms=5000)
            self.led_picker.update_mu(cfg.rwd_density, mu_nr)
        elif stage == 3:
            led_ms = self._difficulty.led_ms or int(cfg.staircase_start)
            self._difficulty = Difficulty(mu_r=cfg.rwd_density,
                                          mu_nr=cfg.no_rwd_density,
                                          led_ms=led_ms)
            self.led_picker.update_mu(cfg.rwd_density, cfg.no_rwd_density)
        elif stage == 4:
            mu_nr = max(self._difficulty.mu_nr, self._checkpoint_floor)
            self._difficulty = Difficulty(mu_r=cfg.rwd_density,
                                          mu_nr=mu_nr,
                                          led_ms=int(self.settings.min_tower_duration))
            self.led_picker.update_mu(cfg.rwd_density, mu_nr)
        elif stage == 5:
            self._difficulty = Difficulty(
                mu_r=cfg.rwd_density, mu_nr=cfg.no_rwd_density,
                led_ms=int(self.settings.min_tower_duration))
            self.led_picker.update_mu(cfg.rwd_density, cfg.no_rwd_density)

        print(f"   * _apply_stage {stage}: {cfg.name} "
              f"(mu_nr={self._difficulty.mu_nr:.3f}, "
              f"led_ms={self._difficulty.led_ms}ms)")

    def _pass_checkpoint(self, to_stage: int) -> None:
        self._checkpoint = to_stage - 1
        new_start = STAGES[to_stage].staircase_start
        self._checkpoint_floor = new_start

        if to_stage == 3:
            self._difficulty = Difficulty(led_ms=int(new_start))
        elif to_stage in (2, 4):
            self._difficulty = Difficulty(mu_nr=new_start)
        else:
            self._difficulty = Difficulty()
        self._perf_window = deque(maxlen=int(self.settings.acc_window))
        self._rescue_trials_left = 0
        self._streak = 0
        self._main_trial = 0
        self._apply_stage(to_stage)
        self.settings.checkpoint = self._checkpoint
        self.settings.checkpoint_floor = self._checkpoint_floor
        self.settings.stage = to_stage
        print(f"   * Checkpoint {self._checkpoint} passed!"
              f" -> Stage {to_stage} (floor={self._checkpoint_floor:.3f})")

    def _after_trial_adaptation(self) -> None:
        """Update difficulty parameters based on trial outcome,
        and check for checkpoint advancement."""
        if self.is_trial_correct is None:
            return
        correct = bool(self.is_trial_correct)
        stage = self._current_stage
        cfg = STAGES[stage]

        self._perf_window.append(int(correct))

        self._last_delta = 0.0
        self._last_boost = 1.0

        # Warmup phase: easy one-sided (mu_nr=0), no staircase update
        if self._phase == "warmup":
            self._warmup_perf.append((self.current_trial_rwd_side, correct))
            n = len(self._warmup_perf)
            min_t = int(self.settings.warmup_min_trials)
            if n >= min_t:
                wm_acc = sum(c for _, c in self._warmup_perf) / n
                left_c = [c for s, c in self._warmup_perf if s == TrialSide.LEFT]
                right_c = [c for s, c in self._warmup_perf
                           if s == TrialSide.RIGHT]
                wm_bias = (abs(sum(left_c) / len(left_c)
                               - sum(right_c) / len(right_c))
                           if left_c and right_c else 1.0)
                if (wm_acc >= float(self.settings.warmup_acc_threshold)
                        and wm_bias <= float(self.settings.warmup_bias_threshold)):
                    self._phase = "main"
                    self._main_trial = 0
                    self.led_picker.update_mu(cfg.rwd_density,
                                             self._difficulty.mu_nr)
                    print(f"   * Warmup passed! n={n}, "
                          f"acc={wm_acc:.0%}, bias={wm_bias:.0%}")
            return

        # Stages without trial-by-trial staircase (S0, S1, S5 Final)
        if cfg.staircase_variable == "none":
            if stage == 5:  # Final stage: rescue block with easy trials
                if self._rescue_trials_left > 0:
                    self._rescue_trials_left -= 1
                    if self._rescue_trials_left == 0:
                        self.led_picker.update_mu(cfg.rwd_density,
                                                  self._difficulty.mu_nr)
                        print("   * Rescue complete, returning to main task")
                    return
                if (self.settings.rescue_enabled
                        and len(self._perf_window) >= self.settings.acc_window
                        and (sum(self._perf_window) / len(self._perf_window))
                        < self.settings.rescue_threshold):
                    self._rescue_trials_left = int(
                        self.settings.rescue_block_size)
                    self.led_picker.update_mu(cfg.rwd_density, 0.0)
                    print(f"   * Rescue triggered! "
                          f"{self.settings.rescue_block_size} easy trials")
                return
            if (len(self._perf_window) >= self.settings.acc_window
                    and stage == 1
                    and self.settings.stage <= MAX_STAGE):
                rolling_acc = sum(self._perf_window) / len(self._perf_window)
                bias = abs(self.left_or_right.current_empR - 0.5)
                if rolling_acc >= cfg.advance_threshold and bias <= 0.10:
                    self._pass_checkpoint(to_stage=2)
            return

        # Staircase stages (S2, S3, S4): main phase only (warmup handled above)
        self._main_trial += 1
        delta, self._streak, self._last_boost = cfg.compute_step(
            correct, self._streak, self._main_trial, self.settings)
        self._last_delta = delta

        if cfg.staircase_harder_direction == "up":
            if correct:  # increase difficulty if correct, but not above target
                self._difficulty.mu_nr = min(
                    self._difficulty.mu_nr + delta, cfg.staircase_target)
            else:  # decrease difficulty if incorrect, but not below checkpoint
                self._difficulty.mu_nr = max(
                    self._difficulty.mu_nr - delta, self._checkpoint_floor)
            self.led_picker.update_mu(cfg.rwd_density, self._difficulty.mu_nr)

        elif cfg.staircase_harder_direction == "down":
            # same for stage 3: but led_ms decreases when harder
            if correct:
                self._difficulty.led_ms = max(
                    self._difficulty.led_ms - int(delta),
                    int(self.settings.min_tower_duration))
            else:
                self._difficulty.led_ms = min(
                    self._difficulty.led_ms + int(delta),
                    int(self._checkpoint_floor))

        # Checkpoint assessment
        if self.settings.stage > MAX_STAGE:  # done
            return

        if len(self._perf_window) >= self.settings.acc_window:
            rolling_acc = sum(self._perf_window) / len(self._perf_window)

            if (stage == 2
                    and rolling_acc >= cfg.advance_threshold
                    and self._difficulty.mu_nr >= cfg.staircase_target):
                self._pass_checkpoint(to_stage=3)
            elif (stage == 3
                  and rolling_acc >= cfg.advance_threshold
                  and self._difficulty.led_ms
                  <= self.settings.min_tower_duration):
                self._pass_checkpoint(to_stage=4)
            elif (stage == 4
                  and rolling_acc >= cfg.advance_threshold
                  and self._difficulty.mu_nr >= cfg.staircase_target):
                self._pass_checkpoint(to_stage=5)

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [55, 220, 585, 258, 65])
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "BLACK")

    def start(self):
        super().start()
        self.load_led_calibration()

        self.left_valve_opening_time = 1  # self.water_calibration.get_valve_time(
        #     port=1, volume=self.settings.big_reward_amount_ml
        # )
        self.middle_valve_opening_time = .250  # self.water_calibration.get_valve_time(
        #     port=2, volume=self.settings.small_reward_amount_ml
        # )
        self.right_valve_opening_time = 1  # self.water_calibration.get_valve_time(
        #     port=3, volume=self.settings.big_reward_amount_ml
        # )
        self.settings.iti_time = 0
        self.settings.response_time = 60

        self._current_stage = min(int(getattr(self.settings, "stage", 0)),
                                  MAX_STAGE)
        self._checkpoint = int(getattr(self.settings, "checkpoint", 0))
        self._checkpoint_floor = float(getattr(self.settings,
                                               "checkpoint_floor", 0.0))
        floor = self._checkpoint_floor
        resume = bool(getattr(self.settings, "resume_from_last", True))
        if self._current_stage in (2, 4):
            if resume:
                last = float(getattr(self.settings, "last_mu_nr", floor))
                self._difficulty = Difficulty(mu_nr=max(last, floor))
            else:
                self._difficulty = Difficulty(mu_nr=floor)
        elif self._current_stage == 3:
            if resume:
                last_ms = int(getattr(self.settings, "last_led_ms", 5000))
                self._difficulty = Difficulty(led_ms=last_ms if last_ms else 5000)
            else:
                self._difficulty = Difficulty(led_ms=int(floor) if floor else 5000)
        else:
            self._difficulty = Difficulty()
        self._streak = 0
        self._warmup_perf = deque(maxlen=int(self.settings.warmup_min_trials))
        self._main_trial = 0
        self._perf_window = deque(maxlen=int(self.settings.acc_window))
        self._rescue_trials_left = 0
        cfg = STAGES[self._current_stage]
        self._phase = "warmup" if cfg.has_warmup else "main"
        self.led_picker = LedPicker(
            rwd_density=0.0, no_rwd_density=0.0,
            start_dead_zone_cm=self.settings.led_start_dead_zone_cm)
        self._apply_stage(self._current_stage)
        if self._phase == "warmup":
            # Override led_picker to easy one-sided (mu_nr=0) until gate met
            self.led_picker.update_mu(cfg.rwd_density, 0.0)
            print(f"   * Session warmup: mu_nr=0 until "
                  f">={int(self.settings.warmup_min_trials)} trials at "
                  f">={self.settings.warmup_acc_threshold:.0%} acc, "
                  f"<={self.settings.warmup_bias_threshold:.0%} bias")
        self._update_hud()

    def get_LEDs_for_trial(self, verbose=True):
        # Draw trial side (left or right).
        self.current_trial_rwd_side = self.left_or_right.draw_next_trial()

        # Draw LED positions for this trial.
        rwd_leds, no_rwd_leds = self.led_picker.draw_towers()

        if len(rwd_leds) < len(no_rwd_leds):
            # Switch so that rewarded side always has more than n_rewarded
            rwd_leds, no_rwd_leds = no_rwd_leds, rwd_leds

        # Allocate drawn LEDs to both sides and map to physical strip indices
        if self.current_trial_rwd_side == TrialSide.RIGHT:
            map_rwd_leds = self._to_strip_indices(rwd_leds,
                                                  TrialSide.RIGHT)
            map_no_rwd_leds = self._to_strip_indices(no_rwd_leds,
                                                     TrialSide.LEFT)
            self._this_trial_leds[TrialSide.RIGHT] = rwd_leds
            self._this_trial_leds[TrialSide.LEFT] = no_rwd_leds
        elif self.current_trial_rwd_side == TrialSide.LEFT:
            map_rwd_leds = self._to_strip_indices(rwd_leds,
                                                  TrialSide.LEFT)
            map_no_rwd_leds = self._to_strip_indices(no_rwd_leds,
                                                     TrialSide.RIGHT)
            self._this_trial_leds[TrialSide.LEFT] = rwd_leds
            self._this_trial_leds[TrialSide.RIGHT] = no_rwd_leds
        else:
            raise ValueError(f"Invalid trial: {self.current_trial_rwd_side}")

        self.available_leds_idx.update(map_rwd_leds)
        self.available_leds_idx.update(map_no_rwd_leds)
        if verbose:
            print(f"   * Trial {self.current_trial_rwd_side.value}")
            print(f"     - {len(rwd_leds)} RWD LEDs idx:{rwd_leds} "
                  f"--> LED map: {map_rwd_leds}")
            print(f"     - {len(no_rwd_leds)} NO RWD LEDs idx:{no_rwd_leds} "
                  f"--> LED map: {map_no_rwd_leds}")

        self._build_led_triggers()
        self._build_led_pos()
        self.debug_color_state_leds()

    def _build_led_pos(self):
        self.cam_box.items_to_draw["led_pos"] = (
            [self.led_positions[i] for i in self.available_leds_idx]
            )

    def _build_led_triggers(self):
        """Pre-compute sorted (trigger_x, led_idx) list
        from current available_leds_idx."""
        self.led_triggers = sorted(
            [(self.led_positions[i].x_hat + self.distance_offset, int(i))
             for i in self.available_leds_idx],
            reverse=True
        )
        self.next_trigger = self.led_triggers[0][0] if self.led_triggers else 0
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger

    def softcode_callback(self):
        """Dispatch camera_callback based on current task parameters."""
        if self.current_x is not None and self.current_y is not None:
            self.animal_trace.append((int(self.current_x),
                                      int(self.current_y)))
            self.cam_box.items_to_draw["animal_trace"] = self.animal_trace

        self.debug_color_state_leds()

        if STAGES[self._current_stage].timed_leds:
            self._softcode_callback_proximity()
        elif self._current_stage > 0:
            self._softcode_callback_always_on()

    def _softcode_callback_always_on(self):
        """Light all trial LEDs at once and leave them on (stages 1-2)."""
        if not self.available_leds_idx:
            return
        leds = list(self.available_leds_idx)
        self.current_led = leds[0] if len(leds) == 1 else leds
        manager.run_softcode_function(self.SOFTCODE_ALL_LEDS_ON)
        self.used_leds_idx.update(self.available_leds_idx)
        self.available_leds_idx.clear()
        print(f"[LED] always-on: lit {leds}")

    def _softcode_callback_proximity(self):
        """Trigger LEDs one by one as animal passes them (stages 3-4)."""
        # If no LED triggers, do nothing
        if not self.led_triggers:
            return

        # If current position hasn't passed furthest_x, do nothing
        if self.current_x >= self._furthest_x:
            return

        # Update furthest_x and display it
        self._furthest_x = self.current_x
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x

        # If furthest_x hasn't passed next_trigger, do nothing
        if self._furthest_x > self.next_trigger:
            return

        # Collect all LEDs whose trigger has been crossed, then fire at once
        triggered = []
        while (self.led_triggers and
               self._furthest_x <= self.led_triggers[0][0]):
            trigger_x, led_idx = self.led_triggers.pop(0)
            triggered.append(led_idx)
            self.available_leds_idx.discard(led_idx)
            self.used_leds_idx.add(led_idx)

        if triggered:
            self.current_led = (triggered[0] if len(triggered) == 1
                                else triggered)
            manager.run_softcode_function(self.SOFTCODE_SINGLE_LED_ON)
            print("+", self.available_leds_idx, ">>>", triggered, ">>>  -",
                  self.used_leds_idx, "::: trigger=", trigger_x)

        # Update next_trigger to next LED trigger and display it
        if self.led_triggers:
            self.next_trigger = self.led_triggers[0][0]
        else:
            self.next_trigger = 641
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger

    def debug_color_state_leds(self):
        return
        for idx in range(self.led_strip.num_leds):
            color = self.COLOR_UNSELECTED
            if idx in self.available_leds_idx:
                color = self.COLOR_SELECTED
            if idx in self.used_leds_idx:
                color = self.COLOR_USED
            self.led_strip.set_led_color(idx, *color)
        self.led_strip.update_strip(sleep_duration=None)

    def create_trial(self):
        if self.give_free_reward:
            self.middle_poke_action = "SMALL REWARD"
        else:
            self.middle_poke_action = "END TRIAL"

        self.cues = []
        if STAGES[self._current_stage].both_sides_rewarded:
            # Stage 0: motor routine, both ports rewarded
            self.current_trial_rwd_side = TrialSide.BOTH
            left_outputs = [Output.Valve1]
            left_opening_time = self.left_valve_opening_time
            right_outputs = [Output.Valve3]
            right_opening_time = self.right_valve_opening_time
        else:
            # Stages 1 to 3: only the rewarded side opens its valve
            self.get_LEDs_for_trial()
            if self.current_trial_rwd_side == TrialSide.LEFT:
                left_outputs = [Output.Valve1]
                left_opening_time = self.left_valve_opening_time
                right_outputs = []
                right_opening_time = 0
                if self.trial_is_cued:
                    self.cues.append((Output.PWM1,
                                      self.settings.light_intensity_high))
                    self.cues.append((Output.PWM3,
                                      self.settings.light_intensity_low))
            elif self.current_trial_rwd_side == TrialSide.RIGHT:
                right_outputs = [Output.Valve3]
                right_opening_time = self.right_valve_opening_time
                left_outputs = []
                left_opening_time = 0
                if self.trial_is_cued:
                    self.cues.append((Output.PWM3,
                                      self.settings.light_intensity_high))
                    self.cues.append((Output.PWM1,
                                      self.settings.light_intensity_low))
            else:
                raise ValueError(
                    f"Invalid trial side: {self.current_trial_rwd_side}")

        self.bpod.add_state(
            state_name="START TRIAL",
            state_timer=0,
            state_change_conditions={Event.Tup: "WAIT FOR CHOICE POKE"},
            output_actions=[]
        )

        self.bpod.add_state(
            state_name="WAIT FOR CHOICE POKE",
            state_timer=self.settings.response_time,
            state_change_conditions={Event.Tup: "END TRIAL",
                                     Event.Port1In: "POKE LEFT",
                                     Event.Port3In: "POKE RIGHT"},
            output_actions=[("SoftCode", self.SOFTCODE_CAMERA_ACCEPT),
                            *self.cues],
        )

        timed = STAGES[self._current_stage].timed_leds
        poke_softcodes = ([("SoftCode", self.SOFTCODE_CAMERA_REFUSE)]
                          if timed else
                          [("SoftCode", self.SOFTCODE_CAMERA_REFUSE),
                           ("SoftCode", self.SOFTCODE_LED_OFF)])

        self.bpod.add_state(
            state_name="POKE LEFT",
            state_timer=left_opening_time,
            state_change_conditions={Event.Tup: "POKE MIDDLE"},
            output_actions=[*left_outputs, *poke_softcodes],
        )

        self.bpod.add_state(
            state_name="POKE RIGHT",
            state_timer=right_opening_time,
            state_change_conditions={Event.Tup: "POKE MIDDLE"},
            output_actions=[*right_outputs, *poke_softcodes],
        )

        self.bpod.add_state(
            state_name="POKE MIDDLE",
            state_timer=0,
            state_change_conditions={Event.Port2In: self.middle_poke_action},
            output_actions=[(Output.PWM2, self.settings.light_intensity_high)])

        self.bpod.add_state(
            state_name="SMALL REWARD",
            state_timer=self.middle_valve_opening_time,
            state_change_conditions={Event.Tup: "END TRIAL"},
            output_actions=[Output.Valve2]
        )

        self.bpod.add_state(
            state_name="END TRIAL",
            state_timer=self.settings.iti_time,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", self.SOFTCODE_LED_OFF)],
        )

    def current_trial_is_correct(self) -> bool | None:
        """Determine if the trial was correct based on the first poke."""
        if self.current_trial_rwd_side == TrialSide.NONE:
            return None
        if self.current_trial_rwd_side == TrialSide.BOTH:
            return True

        port_to_side = {"Port1In": TrialSide.LEFT,
                        "Port3In": TrialSide.RIGHT}
        first_poke = next((e for e in self.trial_data["ordered_list_of_events"]
                           if e in port_to_side), None)
        return port_to_side.get(first_poke) == self.current_trial_rwd_side

    def after_trial(self):
        self.register_value(f"{TrialSide.LEFT.value} LEDs",
                            self._this_trial_leds[TrialSide.LEFT].tolist())
        self.register_value(f"{TrialSide.RIGHT.value} LEDs",
                            self._this_trial_leds[TrialSide.RIGHT].tolist())

        self.register_value("trial_side", self.current_trial_rwd_side.value)
        self.register_value("water", self.settings.reward_amount_ml)

        self.is_trial_correct = self.current_trial_is_correct()
        self.register_value("trial_correct", self.is_trial_correct)

        # LED picker info
        self.register_value("rwd_density", self.led_picker.mu_reward)
        self.register_value("no_rwd_density", self.led_picker.mu_no_reward)

        # Left or Right trial info
        self.register_value("pR", self.left_or_right.current_PR)
        self.register_value("empR", self.left_or_right.current_empR)
        self.register_value("draw_side",
                            self.left_or_right.current_draw_side.value)
        self.register_value("draw_prob", self.left_or_right.current_draw_prob)

        # Difficulty info
        self.register_value("stage", self._current_stage)
        self.register_value("phase", self._phase)
        self.register_value("mu_r", self._difficulty.mu_r)
        self.register_value("mu_nr", 0.0 if self._phase == "warmup"
                            else self._difficulty.mu_nr)
        self.register_value("led_ms", self._difficulty.led_ms)
        self.register_value("checkpoint_floor", self._checkpoint_floor)
        self.register_value("streak", self._streak)
        self.register_value("checkpoint", self._checkpoint)
        self.register_value("warmup_trial", len(self._warmup_perf))
        self.register_value("trial_is_cued", int(self.trial_is_cued))
        self.register_value("give_free_reward", int(self.give_free_reward))
        self.register_value("rescue", int(self._rescue_trials_left > 0))
        if self.current_trial_rwd_side in (TrialSide.LEFT, TrialSide.RIGHT):
            other_side = (TrialSide.RIGHT
                          if self.current_trial_rwd_side == TrialSide.LEFT
                          else TrialSide.LEFT)
            n_rwd = len(self._this_trial_leds[self.current_trial_rwd_side])
            n_nrwd = len(self._this_trial_leds[other_side])
            self.register_value("delta_towers", n_rwd - n_nrwd)
        else:
            self.register_value("delta_towers", 0)

        print(f"Trial result: side={self.current_trial_rwd_side.value}, "
              f"correct={self.is_trial_correct}, "
              f"leds={self._this_trial_leds}")

        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []
        self.next_trigger = 641
        self._furthest_x = 641
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger
        self.cam_box.items_to_draw["led_pos"] = []
        self.animal_trace = deque(maxlen=25 * 5)
        self.cam_box.items_to_draw["animal_trace"] = self.animal_trace
        if self.is_trial_correct is not None:
            self._after_trial_adaptation()
        self.register_value("step_delta", self._last_delta)
        self.register_value("step_boost", self._last_boost)
        self._update_hud()
        self.left_or_right.add_trial(
            TrialResult(side=self.current_trial_rwd_side,
                        correct=self.is_trial_correct))
        self.current_trial_rwd_side = TrialSide.NONE
        self.is_trial_correct = None

    def close(self):
        self._close_strip()
