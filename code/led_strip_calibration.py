
from pathlib import Path
import matplotlib.pyplot as plt
from village.custom_classes.task import Event
from village.settings import settings
from tower_task_base import TowersTaskBase

# Task to sequentially turn ON each LED in the strip, capture (with cam_box)
# the detected (x, y) position, then turn it OFF and move to the next LED.
# The result is a mapping of led_index -> (x, y) position in the camera's view.
# This gives us the LED positions in camera coordinates, which we will
# certainly use in the main Task.


class LedStripCalibration(TowersTaskBase):
    """
    Sequentially turn ON each LED in the strip, capture (with cam_box) the
    detected (x, y) position, then turn it OFF and move to the next LED.
    The result is a mapping of led_index -> (x, y) position from camera POV.
    This gives us the LED positions in camera coordinates, which we will
    certainly use in the main Task at some point.
    """
    def __init__(self):
        super().__init__()
        self.on_time = 0.4  # LED ON duration (s) so camera can detect
        self.off_time = 0.05  # duration (s) between LEDs
        self.settle_time = 0.05  # duration (s) between LED ON and frame cap

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [60, 285, 560, 292, 200])  # L, T, R, B, Thr
        settings.set("AREA2_BOX", [60, 311, 560, 318, 200])
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "ALLOWED")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "WHITE")

    def start(self):
        super().start()
        self.maximum_number_of_trials = self.led_strip.num_leds

    def create_trial(self):
        self.bpod.add_state(
            state_name="LED_ON",
            state_timer=self.settle_time + self.on_time + self.off_time,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", self.SOFTCODE_LED_ON_CAPTURE)],
        )

    def softcode_callback(self):
        current_led = self.led_positions[self.current_led]
        current_led.add_sample(self.current_x, self.current_y)

    def after_trial(self):
        self.current_led += 1

    def close(self):
        self._close_strip()
        self.save_led_calibration()
        self.make_plot()

    def make_plot(self):
        _, ax = plt.subplots(figsize=(5, 5))
        for i in range(self.led_strip.num_leds):
            x, y = self.led_positions[i].x_hat, self.led_positions[i].y_hat
            ax.scatter(x, y)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        path = Path(settings.get("DATA_DIRECTORY"),
                    self.CALIBRATION_PATH + ".png")
        plt.savefig(path)
