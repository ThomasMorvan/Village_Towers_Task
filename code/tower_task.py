
from typing import Literal
import numpy as np
from village.custom_classes.task import  Event
from village.settings import settings
from village.manager import manager
from tower_task_base import TowersTaskBase, LEDPosition


# task


class TowersTask(TowersTaskBase):
    """
    
    """

    def __init__(self):
        super().__init__()
        self.on_state_duration = 300
        self.led_on_duration = 0.2  # Led on in (s)
        self._furthest_x = -1
        self.available_leds_idx = set()
        self.used_leds_idx = set()
        self.next_trigger = 0
        self.distance_offset = 20  # distance in front of centroid in pixels for now
        self.mock_x = 0

    def set_ui_settings(self):
        settings.set("AREA2_BOX", [0, 40, 640, 70, 120])
        settings.set("USAGE1_BOX", "OFF")
        settings.set("USAGE2_BOX", "ALLOWED")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "BLACK")

    def start(self):
        super().start()
        self.maximum_number_of_trials = 10
        self.load_led_calibration()

    def select_leds_idx(self):
        N = 10
        l = range(self.led_strip.num_leds)
        l = list(range(20)) + list(range(25, self.led_strip.num_leds))
        selected = np.random.choice(l, size=N, replace=False)
        self.available_leds_idx.update([int(i) for i in selected])
        self.debug_color_state_leds()

    def get_next_LED(self, available_leds_idx: list[int],
                     axis: Literal["x", "y"] = "x"):
        if axis == "x":
            next_idx = min(available_leds_idx, key=lambda i: self.led_positions[i].x_hat)
            self.next_trigger = self.led_positions[next_idx].x_hat - self.distance_offset
        elif axis == "y":
            next_idx = min(available_leds_idx, key=lambda i: self.led_positions[i].y_hat)
            self.next_trigger = self.led_positions[next_idx].y_hat - self.distance_offset
        else:
            raise NotImplementedError

        return int(next_idx)

    def softcode_callback(self, auto=True):
        self.mock_x += 1
        self.mock_x = min(self.mock_x, 640)

        self.debug_color_state_leds()
        best_idx = -1
        if len(self.available_leds_idx) == 0:
            return

        if self._furthest_x > self.next_trigger:
            self._furthest_x = 0

        if auto:
            if self.mock_x <= self._furthest_x:
                return
            self._furthest_x = self.mock_x
        else:
            if self.current_x <= self._furthest_x:
                return
            self._furthest_x = self.current_x


        if self._furthest_x < self.next_trigger:
            return

        best_idx = self.get_next_LED(self.available_leds_idx)

        if best_idx == -1:
            return

        self.current_led = best_idx
        manager.run_softcode_function(self.SOFTCODE_SINGLE_LED_ON)


        self.available_leds_idx.remove(best_idx)
        self.used_leds_idx.add(best_idx)

        print("+", self.available_leds_idx, ">>>", best_idx, ">>>  -", self.used_leds_idx, "::: ", self.next_trigger)

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
            state_change_conditions={Event.Tup: "CAMERA_OFF"},
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
        self.next_trigger = 0

    def close(self):
        self._close_strip()
