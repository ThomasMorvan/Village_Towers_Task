
from collections import deque
import numpy as np
import json
from village.custom_classes.task import Event
from village.settings import settings
from village.manager import manager
from tower_task_base import TowersTaskBase
from left_or_right import LeftOrRight, TrialSide, TrialResult
from LEDpicker import LedPicker

# TODO:
# "istrialcorrect"


class TowersTask(TowersTaskBase):
    """Towers Task."""

    def __init__(self):
        super().__init__()
        self.on_state_duration = 300
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

        self.REWARDED_DENSITY = 7.7  # TODO: add in settings
        self.NO_REWARD_DENSITY = 2.3  # TODO: add in settings
        self.led_picker = LedPicker(rwd_density=self.REWARDED_DENSITY,
                                    no_rwd_density=self.NO_REWARD_DENSITY)

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

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [75, 290, 555, 320, 65])
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "BLACK")

    def start(self):
        super().start()
        self.maximum_number_of_trials = 10
        self.load_led_calibration()

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
        self.cam_box.items_to_draw["led_pos"] = sorted(
            [self.led_positions[i].x_hat
                for i in self.available_leds_idx]
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

    def softcode_callback(self):
        if self.current_x is not None and self.current_y is not None:
            self.animal_trace.append((int(self.current_x),
                                      int(self.current_y)))
            self.cam_box.items_to_draw["animal_trace"] = self.animal_trace

        self.debug_color_state_leds()

        # If no more LED triggers, do nothing
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

        # Fire all LEDs whose trigger has been crossed
        while (self.led_triggers and
               self._furthest_x <= self.led_triggers[0][0]):
            trigger_x, led_idx = self.led_triggers.pop(0)
            self.current_led = led_idx
            manager.run_softcode_function(self.SOFTCODE_SINGLE_LED_ON)
            self.available_leds_idx.discard(led_idx)
            self.used_leds_idx.add(led_idx)
            print("+", self.available_leds_idx, ">>>", led_idx, ">>>  -",
                  self.used_leds_idx, "::: trigger=", trigger_x)

        # Update next_trigger to next LED trigger and display it
        if self.led_triggers:
            self.next_trigger = self.led_triggers[0][0]
        else:
            self.next_trigger = 641
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger

    def debug_color_state_leds(self):
        for idx in range(self.led_strip.num_leds):
            color = self.COLOR_UNSELECTED
            if idx in self.available_leds_idx:
                color = self.COLOR_SELECTED
            if idx in self.used_leds_idx:
                color = self.COLOR_USED
            self.led_strip.set_led_color(idx, *color)
        self.led_strip.update_strip(sleep_duration=None)

    def create_trial(self):
        self.get_LEDs_for_trial()
        self.bpod.add_state(
            state_name="TASK_ON",
            state_timer=0,
            state_change_conditions={Event.Tup: "CAMERA_ON"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="CAMERA_ON",
            state_timer=self.on_state_duration,
            state_change_conditions={Event.Tup: "CAMERA_OFF",
                                     Event.Port2In: 'CAMERA_OFF',
                                     },
            output_actions=[("SoftCode", self.SOFTCODE_CAMERA_ACCEPT)],
        )

        self.bpod.add_state(
            state_name="CAMERA_OFF",
            state_timer=0,
            state_change_conditions={Event.Tup: "EXIT"},
            output_actions=[("SoftCode", self.SOFTCODE_CAMERA_REFUSE)],
        )

        self.bpod.add_state(
            state_name="EXIT",
            state_timer=1,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", self.SOFTCODE_LED_OFF)],
        )

    def current_trial_is_correct(self) -> bool | None:
        """Determine if the trial was correct based on the first poke."""
        if self.current_trial_rwd_side == TrialSide.NONE:
            return None

        port_to_side = {"Port1In": TrialSide.LEFT,
                        "Port3In": TrialSide.RIGHT}
        first_poke = next((e for e in self.trial_data["ordered_list_of_events"]
                           if e in port_to_side), None)
        return port_to_side.get(first_poke) == self.current_trial_rwd_side

    def after_trial(self):
        self.register_value("trial_leds",
                            json.dumps({TrialSide.LEFT.value:
                                        self._this_trial_leds[TrialSide.LEFT],
                                        TrialSide.RIGHT.value:
                                        self._this_trial_leds[TrialSide.RIGHT]
                                        }))
        self.register_value("trial_side", self.current_trial_rwd_side.value)
        self.register_value("water", self.settings.reward_amount_ml)

        self.is_trial_correct = self.current_trial_is_correct()
        self.register_value("trial_correct", self.is_trial_correct)

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
        self.left_or_right.add_trial(
            TrialResult(side=self.current_trial_rwd_side,
                        correct=self.is_trial_correct)
                        )
        self.current_trial_rwd_side = TrialSide.NONE
        self.is_trial_correct = None

    def close(self):
        self._close_strip()
