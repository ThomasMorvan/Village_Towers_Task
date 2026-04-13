
import numpy as np
from village.custom_classes.task import Event
from village.settings import settings
from tower_task_base import TowersTaskBase, LEDPosition


# Task to track any object with cam_box and turn the closest LED
# using the calibration done in led_strip_calibration.py.
# The Light Follows.

class TheLightFollows(TowersTaskBase):
    """
    Track any object and light the closest LED
    """

    def __init__(self):
        super().__init__()
        self.process_every = 5
        self.on_state_duration = 300

    def start(self):
        super().start()
        self.maximum_number_of_trials = 1
        self.load_led_calibration()

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [0, 70, 640, 475, 180])
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "WHITE")

    def faster_distance_to_LED(self, led: LEDPosition):
        # skip sqrt for faster distance comparison
        dx = led.x_hat - self.current_x
        dy = led.y_hat - self.current_y
        return dx * dx + dy * dy

    def softcode_callback(self):
        if not self.process_every == 0:
            self.process_every -= 1
            return

        best_idx = -1
        best_distance = np.inf
        for k, led in self.led_positions.items():
            distance = self.faster_distance_to_LED(led=led)
            if distance < best_distance:
                best_idx = int(k)
                best_distance = distance
        print(best_idx, best_distance)

        if best_idx == -1:
            return

        self.led_strip.clear_strip()
        self.led_strip.set_led_color(best_idx, 255, 0, 0)
        self.led_strip.update_strip()

        self.process_every = 5

    def create_trial(self):
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
        pass

    def close(self):
        self._close_strip()
