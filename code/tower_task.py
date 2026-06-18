
from collections import deque
import numpy as np
from village.custom_classes.task_base import (BpodEvent as Event,
                                              BpodOutput as Output)
from village.settings import settings
from village.scripts.log import log
from tower_task_base import TowersTaskBase
from left_or_right import LeftOrRight, TrialSide, TrialResult
from LEDpicker import LedPicker
from reward_policy import RewardPolicy
from task_stages import STAGES
from online_difficulty_controller import (OnlineDifficultyController,
                                          AdaptationEvent)


class TowersTask(TowersTaskBase):
    """Towers Task."""

    def __init__(self):
        super().__init__()
        self._furthest_x = settings.get("CAM_BOX_RESOLUTION")[0] + 1
        self._this_trial_leds = {TrialSide.LEFT: np.array([], dtype=int),
                                 TrialSide.RIGHT: np.array([], dtype=int)}
        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []  # sorted list of (trigger_x, led_idx)
        self.next_trigger = settings.get("CAM_BOX_RESOLUTION")[0] + 1
        self.distance_offset = 50  # FIXME: px in front of centroid for trigger
        self.left_or_right = LeftOrRight()
        self.current_trial_rwd_side: TrialSide = TrialSide.NONE
        self.is_trial_correct: bool = None
        self.animal_trace = deque(maxlen=25*5)

        # Placeholder, recreated in start() once settings are loaded
        self.led_picker = LedPicker(rwd_density=0.0, no_rwd_density=0.0)

        self.trial_is_cued = False
        self.give_free_reward = True

        self._odc = OnlineDifficultyController()
        self._reward_policy = RewardPolicy()
        self._reward_mult = 1.0

    def _to_strip_indices(self, leds: np.ndarray, side: TrialSide
                          ) -> np.ndarray:
        """Map LedPicker indices (0-71) to physical strip indices,
        because strip goes from [0] start-right to [71] end-right,
        then [72] end-left to [143] start-left (no gap, 2*72 LEDs total).

        Right side: 0-71 maps directly (entry end = index 0).
        Left side: reversed and shifted (index 0 maps to the entry end of the
        corridor (physical 143), index 71 maps to the far end (physical 72).
          physical = LEFT_SIDE_OFFSET + (NUM_LEDS - 1) - leds
        """
        LEFT_SIDE_OFFSET = 72
        if side == TrialSide.LEFT:
            return LEFT_SIDE_OFFSET + (self.led_picker.NUM_LEDS - 1) - leds
        return leds

    def _update_hud(self) -> None:
        """Update cam_box items to draw"""
        stage = self._odc.stage
        cfg = STAGES[stage]
        rolling_acc = self._odc.rolling_acc
        acc_ok = (rolling_acc is not None
                  and rolling_acc >= cfg.advance_threshold)
        acc_pct = f"{rolling_acc*100:.0f}" if rolling_acc is not None else "?"
        thr_pct = f"{cfg.advance_threshold*100:.0f}"

        if self._odc.phase == "warmup":
            min_t = self._odc._warmup.min_trials
            acc_thr = self._odc._warmup.acc_threshold
            bias_thr = self._odc._warmup.bias_threshold
            adv_label = [
                ("W-trials:", f" {self._odc.warmup_n}/{min_t}",
                 self._odc.warmup_n >= min_t),
                ("W-acc:",
                 f" {self._odc.warmup_acc*100:.0f}/{acc_thr*100:.0f}%",
                 self._odc.warmup_acc >= acc_thr),
                ("W-bias:",
                 f" {self._odc.warmup_bias*100:.0f}/{bias_thr*100:.0f}%",
                 self._odc.warmup_bias <= bias_thr),
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
        elif stage in (2, 4):
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            mu_nr_lbl = ("mu_nr:", (f" {self._odc.difficulty.mu_nr:.2f}/"
                                    f"{cfg.staircase.target:.1f}"),
                         self._odc.difficulty.mu_nr >= cfg.staircase.target)
            adv_label = [acc_lbl, mu_nr_lbl]
        elif stage == 3:
            acc_lbl = ("Acc:", f" {acc_pct}/{thr_pct}%", acc_ok)
            led_ms_lbl = ("LED ms:", (f" {self._odc.difficulty.led_ms:.0f}/"
                                      f"{self.settings.min_tower_duration:.0f}"
                                      ),
                          (self._odc.difficulty.led_ms <=
                          self.settings.min_tower_duration))
            adv_label = [acc_lbl, led_ms_lbl]
        elif stage == 5:
            adv_label = [("", "  Final stage", True)]
        else:
            adv_label = [("", "  Done", True)]

        self.cam_box.items_to_draw["hud"] = {
            "phase":            self._odc.phase,
            "stage":            stage,
            "stage_name":       cfg.name,
            "difficulty":       self._odc.difficulty,
            "checkpoint":       self._odc.checkpoint,
            "checkpoint_floor": self._odc.checkpoint_floor,
            "streak":           self._odc.streak,
            "rolling_acc":      rolling_acc,
            "warmup_trial":     (self._odc.warmup_n, self._odc.warmup_min),
            "adv_label":        adv_label,
        }

    def _apply_stage(self, stage: int) -> None:
        """update LedPicker + trial flags for the given stage.
        Difficulty is already set in _odc before this is called.
        """
        cfg = STAGES[stage]
        self.trial_is_cued = cfg.trial_is_cued
        self.give_free_reward = cfg.give_free_reward

        diff = self._odc.difficulty
        if stage == 0:
            pass  # no LEDs in S0
        else:
            self.led_picker.update_mu(diff.mu_r, diff.mu_nr)

        self.settings.stage = stage
        self.settings.checkpoint = self._odc.checkpoint
        self.settings.checkpoint_floor = self._odc.checkpoint_floor

    def _after_trial_adaptation(self) -> None:
        """Delegate adaptation to OnlineDifficultyController;
        handle device side-effects."""
        if self.is_trial_correct is None:
            return
        correct = bool(self.is_trial_correct)
        bias = abs(self.left_or_right.current_empR - 0.5)

        event: AdaptationEvent = self._odc.after_trial(
            correct, self.current_trial_rwd_side, self.settings, bias=bias)

        if event.warmup_passed:
            self.led_picker.update_mu(self._odc.difficulty.mu_r,
                                      self._odc.difficulty.mu_nr)

        if event.stage_advanced_to is not None:
            self._apply_stage(event.stage_advanced_to)
            if self._odc.phase == "warmup":
                self.led_picker.update_mu(self._odc.difficulty.mu_r, 0.0)

        if event.rescue_triggered:
            self.led_picker.update_mu(self._odc.difficulty.mu_r, 0.0)

        if event.rescue_ended:
            self.led_picker.update_mu(self._odc.difficulty.mu_r,
                                      self._odc.difficulty.mu_nr)

        # Always keep led_picker in sync after staircase update
        if not any([event.warmup_passed, event.stage_advanced_to,
                    event.rescue_triggered, event.rescue_ended]):
            cfg = STAGES[self._odc.stage]
            if cfg.staircase.variable != "none" and self._odc.phase == "main":
                self.led_picker.update_mu(self._odc.difficulty.mu_r,
                                          self._odc.difficulty.mu_nr)

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [55, 225, 585, 265, 65])
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "BLACK")

    def start(self):
        super().start()
        self.load_led_calibration()

        self.left_valve_opening_time = self.water_calibration.get_valve_time(
            port=1, volume=self.settings.big_reward_amount_ml)
        self.middle_valve_opening_time = self.water_calibration.get_valve_time(
            port=2, volume=self.settings.small_reward_amount_ml)
        self.right_valve_opening_time = self.water_calibration.get_valve_time(
            port=3, volume=self.settings.big_reward_amount_ml)
        self.settings.iti_time = 0
        self.settings.response_time = 60

        self._odc.start(self.settings)
        self._reward_policy = RewardPolicy.from_settings(self.settings)
        self.led_picker = LedPicker(
            rwd_density=0.0, no_rwd_density=0.0,
            start_dead_zone_cm=self.settings.led_start_dead_zone_cm)
        self._apply_stage(self._odc.stage)
        if self._odc.phase == "warmup":
            self.led_picker.update_mu(self._odc.difficulty.mu_r, 0.0)
        self._update_hud()

    def get_LEDs_for_trial(self):
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

        self._build_led_triggers()
        self._build_led_pos()
        self.debug_color_state_leds()

    def _build_led_pos(self):
        self._publish_led_pos()

    def _publish_led_pos(self):
        """Publish not-yet-triggered (green) and
        triggered (red) LED positions."""
        self.cam_box.items_to_draw["led_pos"] = (
            [self.led_positions[i] for i in self.available_leds_idx])
        self.cam_box.items_to_draw["led_pos_used"] = (
            [self.led_positions[i] for i in self.used_leds_idx])

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

        try:
            sma = self.bpod.sma
            if "hud" in self.cam_box.items_to_draw:
                self.cam_box.items_to_draw["hud"]["bpod_state"] = (
                    sma.state_names[sma.current_state])
        except (IndexError, AttributeError):
            pass

        self.debug_color_state_leds()

        if self._leds_timed():
            self._softcode_callback_proximity()
        elif self._odc.stage > 0:
            self._softcode_callback_always_on()

    def _leds_timed(self) -> bool:
        """Whether LEDs are timed this trial. Timed stages still run warmup
        and rescue ("easy block") trials untimed (always-on), matching the
        paper's T4/T7-type easy trials."""
        return (STAGES[self._odc.stage].timed_leds
                and self._odc.phase != "warmup"
                and not self._odc.rescue_active)

    def _trial_delta_towers(self) -> int:
        """Compute delta towers."""
        side = self.current_trial_rwd_side
        if side not in (TrialSide.LEFT, TrialSide.RIGHT):
            return 0
        other = (TrialSide.RIGHT if side == TrialSide.LEFT
                 else TrialSide.LEFT)
        return (len(self._this_trial_leds[side])
                - len(self._this_trial_leds[other]))

    def _softcode_callback_always_on(self):
        """Light all trial LEDs at once and leave them on (stages 1-2)."""
        if not self.available_leds_idx:
            return
        leds = list(self.available_leds_idx)
        self.current_led = leds[0] if len(leds) == 1 else leds
        try:
            self.execute_function(self.SOFTCODE_ALL_LEDS_ON)
        except Exception:
            log.error("Error running function "
                      + str(self.SOFTCODE_ALL_LEDS_ON))

        self.used_leds_idx.update(self.available_leds_idx)
        self.available_leds_idx.clear()
        self._publish_led_pos()

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
            _trigger_x, led_idx = self.led_triggers.pop(0)
            triggered.append(led_idx)
            self.available_leds_idx.discard(led_idx)
            self.used_leds_idx.add(led_idx)

        if triggered:
            self.current_led = (triggered[0] if len(triggered) == 1
                                else triggered)
            try:
                self.execute_function(self.SOFTCODE_SINGLE_LED_ON)
            except Exception:
                log.error("Error running function "
                          + str(self.SOFTCODE_SINGLE_LED_ON))
            self._publish_led_pos()

        # Update next_trigger to next LED trigger and display it
        if self.led_triggers:
            self.next_trigger = self.led_triggers[0][0]
        else:
            self.next_trigger = settings.get("CAM_BOX_RESOLUTION")[0] + 1
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
        if STAGES[self._odc.stage].both_sides_rewarded:
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

        side = self.current_trial_rwd_side
        pol = STAGES[self._odc.stage].policy
        main_phase = (self._odc.phase == "main"
                      and not self._odc.rescue_active)
        self._reward_mult = self._reward_policy.reward_mult_for_trial(
            jackpot=pol.jackpot, effort=pol.effort,
            delta_towers=self._trial_delta_towers(),
            single_sided=side in (TrialSide.LEFT, TrialSide.RIGHT),
            main_phase=main_phase)
        left_opening_time *= self._reward_mult
        right_opening_time *= self._reward_mult

        self.bpod.add_state(
            state_name="START TRIAL",
            state_timer=0,
            state_change_conditions={Event.Tup: "WAIT FOR CHOICE POKE"},
            output_actions=[]
        )

        if STAGES[self._odc.stage].both_sides_rewarded:  # no timeout in S0
            choice_timer = 0
            choice_conditions = {Event.Port1In: "POKE LEFT",
                                 Event.Port3In: "POKE RIGHT"}
        else:
            choice_timer = self.settings.response_time
            choice_conditions = {Event.Tup: "END TRIAL",
                                 Event.Port1In: "POKE LEFT",
                                 Event.Port3In: "POKE RIGHT"}
        self.bpod.add_state(
            state_name="WAIT FOR CHOICE POKE",
            state_timer=choice_timer,
            state_change_conditions=choice_conditions,
            output_actions=[("SoftCode", self.SOFTCODE_CAMERA_ACCEPT),
                            *self.cues],
        )

        timed = self._leds_timed()
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

    def _water_delivered_ml(self) -> float:
        """Actual volume delivered: side port (1/3, big or jackpot, only
        if that side was chosen correctly) plus the middle port (2) free
        reward, if one was given."""
        s = self.settings
        if self.current_trial_rwd_side == TrialSide.BOTH:
            side_ml = 2 * s.big_reward_amount_ml
        elif self.is_trial_correct:
            side_ml = (s.jackpot_reward_amount_ml
                       if self._reward_policy.last_was_jackpot
                       else s.big_reward_amount_ml * self._reward_mult)
        else:
            side_ml = 0.0
        evts = self.trial_data["ordered_list_of_events"]
        small_ml = (s.small_reward_amount_ml
                    if (self.give_free_reward and "Port2In" in evts)
                    else 0.0)
        return side_ml + small_ml

    def after_trial(self):
        self.register_value(f"{TrialSide.LEFT.value} LEDs",
                            self._this_trial_leds[TrialSide.LEFT].tolist())
        self.register_value(f"{TrialSide.RIGHT.value} LEDs",
                            self._this_trial_leds[TrialSide.RIGHT].tolist())

        self.register_value("trial_side", self.current_trial_rwd_side.value)
        self.is_trial_correct = self.current_trial_is_correct()
        self.register_value("trial_correct", self.is_trial_correct)
        self.register_value("water", self._water_delivered_ml())
        self.register_value("reward_mult", self._reward_mult)
        self.register_value("jackpot",
                            int(self._reward_policy.last_was_jackpot))

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
        self.register_value("stage", self._odc.stage)
        self.register_value("phase", self._odc.phase)
        self.register_value("mu_r", self._odc.difficulty.mu_r)
        self.register_value("mu_nr",
                            0.0 if self._odc.phase == "warmup"
                            else self._odc.difficulty.mu_nr)
        self.register_value("led_ms", self._odc.difficulty.led_ms)
        self.register_value("checkpoint_floor", self._odc.checkpoint_floor)
        self.register_value("streak", self._odc.streak)
        self.register_value("checkpoint", self._odc.checkpoint)
        self.register_value("warmup_trial", self._odc.warmup_n)
        self.register_value("trial_is_cued", int(self.trial_is_cued))
        self.register_value("give_free_reward", int(self.give_free_reward))
        self.register_value("rescue", int(self._odc.rescue_active))
        self.register_value("delta_towers", self._trial_delta_towers())

        log.info(f"Trial: side={self.current_trial_rwd_side.value}, "
                 f"correct={self.is_trial_correct}")

        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []
        self.next_trigger = settings.get("CAM_BOX_RESOLUTION")[0] + 1
        self._furthest_x = settings.get("CAM_BOX_RESOLUTION")[0] + 1
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger
        self.cam_box.items_to_draw["led_pos"] = []
        self.cam_box.items_to_draw["led_pos_used"] = []
        self.animal_trace = deque(maxlen=25 * 5)
        self.cam_box.items_to_draw["animal_trace"] = self.animal_trace
        if self.is_trial_correct is not None:
            self._after_trial_adaptation()
        self.register_value("step_delta", self._odc.last_delta)
        self.register_value("step_boost", self._odc.last_boost)
        self._update_hud()
        self.left_or_right.add_trial(
            TrialResult(side=self.current_trial_rwd_side,
                        correct=self.is_trial_correct))
        self.current_trial_rwd_side = TrialSide.NONE
        self.is_trial_correct = None

    def close(self):
        self._close_strip()
        # flush overlay
        items = getattr(self.cam_box, "items_to_draw", None)
        if isinstance(items, dict):
            items.clear()
