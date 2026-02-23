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


# Task to sequentially turn ON each LED in the strip, capture (with cam_box)
# the detected (x, y) position, then turn it OFF and move to the next LED.
# The result is a mapping of led_index -> (x, y) position in the camera's view.
# This gives us the LED positions in camera coordinates, which we will
# certainly use in the main Task.

SOFTCODE_LED_OFF = 60
SOFTCODE_LED_BASE = 40

SOFTCODE_CAMERA_ACCEPT = 11
SOFTCODE_CAMERA_REFUSE = 12


class LEDPosition:
    """Simple class to hold LED position data"""
    def __init__(self, idx: int, x_hat=None, y_hat=None):
        self.idx = idx
        self.x_hat = x_hat
        self.y_hat = y_hat
        self.samples_x = []
        self.samples_y = []

    def add_sample(self, x: int, y: int):
        self.samples_x.append(x)
        self.samples_y.append(y)

    def finalize(self):
        if len(self.samples_x) == 0 or len(self.samples_y) == 0:
            self.x_hat = None
            self.y_hat = None
        else:
            self.x_hat = np.median(self.samples_x)
            self.y_hat = np.median(self.samples_y)

    def __repr__(self):
        return f"LED {self.idx}:::({self.x_hat}, {self.y_hat})"


class LedStripCalibration(Task):
    """
    Calibration:
      LED0 ON -> capture -> OFF -> LED1 ON -> capture -> ... -> exit
    Produces: {led_index: {"x": int, "y": int}}
    """

    def __init__(self):
        super().__init__()

        self.on_time = 2  # LED ON duration (s) so camera can detect
        self.off_time = 0.5  # duration (s) between LEDs
        self.settle_time = 0.5  # duration (s) after turning ON before video acquisition

        self._current_led = None
        self._current_x = None
        self._current_y = None
        self._current_frame = None
        self.accept_frames = thEvent()

    def start(self):
        self.led_strip = led_strip
        self.led_positions = {i: LEDPosition(i)
                              for i in range(self.led_strip.num_leds)}
        self.maximum_number_of_trials = 1
        self.reload_softcode()

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

    def blink(self):
        self.led_strip.set_led_color(0, 255, 0, 0)
        self.led_strip.update_strip()
        time.sleep(1)
        self.led_strip.set_led_color(0, 0, 255, 0)
        self.led_strip.update_strip()
        time.sleep(1)
        self.led_strip.set_led_color(0, 0, 0, 255)
        self.led_strip.update_strip()
        time.sleep(1)
        self.led_strip.clear_strip()
        self.led_strip.update_strip()

    @property
    def current_led(self):
        return self._current_led

    @property
    def current_x(self):
        return self._current_x

    @property
    def current_y(self):
        return self._current_y

    @property
    def current_frame(self):
        return self._current_frame

    @current_led.setter
    def current_led(self, i: int):
        """Called from softcode when LED changes,
        so we can track which LED is currently ON."""
        self._current_led = i

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

    def create_trial(self):
        # Idea is Turn LED_i ON for self.on_time, then OFF for self.off_time, then next LED, etc.
        for i in range(self.led_strip.num_leds):
            next_state = f"LED{i+1}_ON" if i < self.led_strip.num_leds - 1 else "EXIT"
            print(f"LED{i}_ON", next_state, ("SoftCode", SOFTCODE_LED_BASE + i))
            self.bpod.add_state(
                state_name=f"LED{i}_ON",
                state_timer=self.settle_time,
                state_change_conditions={Event.Tup: f"CAMERA{i}_ON"},
                output_actions=[("SoftCode", SOFTCODE_LED_BASE + i)],
            )

            self.bpod.add_state(
                state_name=f"CAMERA{i}_ON",
                state_timer=self.on_time,
                state_change_conditions={Event.Tup: f"CAMERA{i}_OFF"},
                output_actions=[("SoftCode", SOFTCODE_CAMERA_ACCEPT)],
            )

            self.bpod.add_state(
                state_name=f"CAMERA{i}_OFF",
                state_timer=0,
                state_change_conditions={Event.Tup: f"LED{i}_OFF"},
                output_actions=[("SoftCode", SOFTCODE_CAMERA_REFUSE)],
            )

            self.bpod.add_state(
                state_name=f"LED{i}_OFF",
                state_timer=self.off_time,
                state_change_conditions={Event.Tup: next_state},
                output_actions=[("SoftCode", SOFTCODE_LED_OFF)],
            )

        self.bpod.add_state(
            state_name="EXIT",
            state_timer=1,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", SOFTCODE_LED_OFF)],
        )

    def add_sample(self):
        current_led = self.led_positions[self.current_led]
        current_led.add_sample(self.current_x, self.current_y)

    def after_trial(self):
        for i in range(self.led_strip.num_leds):
            print(f"LED {i}: samples collected={len(self.led_positions[i].samples_x)}")
            self.led_positions[i].finalize()

        out = {int(k): {"x": int(v.x_hat) if v.x_hat is not None else -1,
                        "y": int(v.y_hat) if v.y_hat is not None else -1}
               for k, v in self.led_positions.items()}
        print(out)
        path = Path(settings.get("DATA_DIRECTORY"), "led_strip_calibration.json")
        path.write_text(json.dumps(out, indent=2))
        self.register_value("led_strip_calibration_path", str(path))
        self.register_value("led_strip_calibration", out)
        self.make_plot()

    def close(self):
        self.accept_frames.clear()
        self.led_strip.clear_strip()
        self.led_strip.update_strip()

    def make_plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for i in range(self.led_strip.num_leds):
            x, y = self.led_positions[i].x_hat, self.led_positions[i].y_hat
            ax.scatter(x, y)
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        path = Path(settings.get("DATA_DIRECTORY"), "led_strip_calibration.png")
        plt.savefig(path)
