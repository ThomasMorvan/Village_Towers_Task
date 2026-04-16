
from typing import Literal
import numpy as np
from village.custom_classes.task import Event
from village.settings import settings
from village.manager import manager
from tower_task_base import TowersTaskBase


class TowersTask(TowersTaskBase):
    """Towers Task."""

    def __init__(self):
        super().__init__()
        self.on_state_duration = 300
        self.led_on_duration = 0.2  # Led on in (s)  # TODO: implement in settings
        self._furthest_x = -1
        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []  # sorted list of (trigger_x, led_idx)
        self.next_trigger = -1   # display only: x position of leds
        self.distance_offset = 50  # FIXME: distance in front of centroid in pixels for now

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

    # TODO: remove and use new methods
    def select_leds_idx(self):
        N = 10
        l = range(self.led_strip.num_leds)
        l = list(range(73)) + list(range(83, self.led_strip.num_leds))
        selected = np.random.choice(l, size=N, replace=False)
        self.available_leds_idx.update([int(i) for i in selected])
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
            [(self.led_positions[i].x_hat - self.distance_offset, int(i))
             for i in self.available_leds_idx]
        )
        self.next_trigger = self.led_triggers[0][0] if self.led_triggers else 0

    def softcode_callback(self):
        self.debug_color_state_leds()

        # If no more LED triggers, do nothing
        if not self.led_triggers:
            return

        # If current position hasn't passed furthest_x, do nothing
        if self.current_x <= self._furthest_x:
            return

        # Update furthest_x and display it
        self._furthest_x = self.current_x
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x

        # If furthest_x hasn't passed next_trigger, do nothing
        if self._furthest_x < self.next_trigger:
            return

        # Fire all LEDs whose trigger has been crossed
        while self.led_triggers and self._furthest_x >= self.led_triggers[0][0]:
            trigger_x, led_idx = self.led_triggers.pop(0)
            self.current_led = led_idx
            manager.run_softcode_function(self.SOFTCODE_SINGLE_LED_ON)
            self.available_leds_idx.discard(led_idx)
            self.used_leds_idx.add(led_idx)
            print("+", self.available_leds_idx, ">>>", led_idx, ">>>  -",
                  self.used_leds_idx, "::: trigger=", trigger_x)

        # Update next_trigger to next LED trigger and display it
        self.next_trigger = self.led_triggers[0][0] if self.led_triggers else self.next_trigger
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
        self.select_leds_idx()
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

    def after_trial(self):
        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.led_triggers = []
        self.next_trigger = -1
        self._furthest_x = -1
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger
        self.cam_box.items_to_draw["led_pos"] = []

    def close(self):
        self._close_strip()
