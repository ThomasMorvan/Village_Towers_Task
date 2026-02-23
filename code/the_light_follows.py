import os
import time
import json
from pathlib import Path
import numpy as np
from threading import Event as thEvent
import importlib
import importlib.util
from village.scripts.log import log
from village.custom_classes.task import Task, Event
from village.devices.led_strip import led_strip
from village.settings import settings
from village.manager import manager
from led_strip_calibration import LEDPosition


# Task to track any object with cam_box and turn the closest LED
# using the calibration done in led_strip_calibration.py.
# The Light Follows.


SOFTCODE_LED_OFF = 60
SOFTCODE_LED_BASE = 40

SOFTCODE_CAMERA_ACCEPT = 11
SOFTCODE_CAMERA_REFUSE = 12


class TheLightFollows(Task):
    """
    Track any object and light the closest LED
    """

    def __init__(self):
        super().__init__()
        self.process_every = 5
        self.on_time = 300
        self._current_x = None
        self._current_y = None
        self._current_frame = None
        self._closest_led = None
        self.accept_frames = thEvent()
        self.led_positions = {}

    @property
    def current_x(self):
        return self._current_x

    @property
    def current_y(self):
        return self._current_y

    @property
    def current_frame(self):
        return self._current_frame

    @current_x.setter
    def current_x(self, x: int):
        """Called from softcode when x changes."""
        self._current_x = x

    @current_y.setter
    def current_y(self, y: int):
        """Called from softcode when y changes."""
        self._current_y = y

    @current_frame.setter
    def current_frame(self, frame_idx: int):
        """Called from softcode when x changes."""
        self._current_frame = frame_idx

    def start(self):
        self.led_strip = led_strip
        self.maximum_number_of_trials = 1
        self.reload_softcode()
        self.load_calibration()
        print(self.led_positions)

    def load_calibration(self):
        path = Path(settings.get("DATA_DIRECTORY"), "led_strip_calibration.json")
        with open(path) as f:
            calibration = json.load(f)

        for k, v in calibration.items():
            x, y = v['x'], v['y']
            self.led_positions[k] = LEDPosition(idx=k, x_hat=x, y_hat=y)

    def distance_to_LED(self, led: LEDPosition):
        dx = np.abs(led.x_hat - self.current_x)
        dy = np.abs(led.y_hat - self.current_y)
        return np.sqrt(dx**2 + dy**2)

    def faster_distance_to_LED(self, led: LEDPosition):
        # skip sqrt for faster distance comparison,
        # we only care about relative distances
        dx = led.x_hat - self.current_x
        dy = led.y_hat - self.current_y
        return dx * dx + dy * dy

    def add_sample(self):  # reuse from calib so same name in softcode
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

    def reload_softcode(self) -> None:
        """Hacky reload softcodes because N_LEDS is initialized dynamically,
        and LED strip is initialized after default softcode load."""
        directory = settings.get("CODE_DIRECTORY")
        functions_path = ""

        for root, _, files in os.walk(directory):
            for file in files:
                if file == "softcode_functions.py":
                    functions_path = os.path.join(root, file)

        if os.path.exists(functions_path):
            module_name = "custom_module"
            spec = importlib.util.spec_from_file_location(module_name, functions_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    for i in range(1, 100):
                        func_name = f"function{i}"
                        if hasattr(module, func_name):
                            manager.functions[i] = getattr(module, func_name)
                except Exception as e:
                    log.error(f"Couldn't reimport softcode functions {e}")

    def create_trial(self):
        self.bpod.add_state(
            state_name="TASK_ON",
            state_timer=0,
            state_change_conditions={Event.Tup: "CAMERA_ON"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="CAMERA_ON",
            state_timer=self.on_time,
            state_change_conditions={Event.Tup: "CAMERA_OFF"},
            output_actions=[("SoftCode", SOFTCODE_CAMERA_ACCEPT)],
        )

        self.bpod.add_state(
            state_name="CAMERA_OFF",
            state_timer=0,
            state_change_conditions={Event.Tup: "EXIT"},
            output_actions=[("SoftCode", SOFTCODE_CAMERA_REFUSE)],
        )

        self.bpod.add_state(
            state_name="EXIT",
            state_timer=1,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", SOFTCODE_LED_OFF)],
        )

    def after_trial(self):
        pass

    def close(self):
        self.accept_frames.clear()
        self.led_strip.clear_strip()
        self.led_strip.update_strip()
