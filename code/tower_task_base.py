import os
import time
import json
from pathlib import Path
import numpy as np
from threading import Event as thEvent
import importlib
import importlib.util
from village.scripts.log import log
from village.custom_classes.task import Task
from village.devices.led_strip import get_led_strip
from village.settings import settings
from village.manager import manager


class Color():
    def __init__(self, red: int = 255, green: int = 255, blue: int = 255):
        self.red = red
        self.green = green
        self.blue = blue

    def __iter__(self):
        yield self.red
        yield self.green
        yield self.blue

    def __repr__(self):
        return f"[{self.red} {self.green} {self.blue}]"


class LEDPosition:
    """Simple class to hold LED position data"""
    def __init__(self, idx: int, x_hat=None, y_hat=None):
        self.idx = int(idx)
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
        return f"LED {self.idx}:::({self.x_hat}, {self.y_hat}) :::" \
               f"{len(self.samples_x)} samples"


class TowersTaskBase(Task):
    SOFTCODE_LED_OFF: int = 0
    SOFTCODE_CAMERA_ACCEPT: int = 1
    SOFTCODE_CAMERA_REFUSE: int = 2
    SOFTCODE_LED_ON_CAPTURE: int = 4
    SOFTCODE_SINGLE_LED_ON: int = 5

    CALIBRATION_PATH: str = "led_strip_calibration"

    COLOR_ON = Color(255, 255, 255)
    COLOR_OFF = Color(0, 0, 0)
    COLOR_SELECTED = Color(0, 10, 0)
    COLOR_UNSELECTED = Color(10, 0, 0)
    COLOR_USED = Color(10, 5, 0)

    NUM_LEDS = 155

    def __init__(self):
        super().__init__()

        self._current_led = 0
        self._current_x = None
        self._current_y = None
        self._current_frame = None
        self.accept_frames = thEvent()
        self.led_on_duration = 0
        self.set_ui_settings()
        self.led_strip = get_led_strip(num_leds=self.NUM_LEDS)
        self.led_positions = {i: LEDPosition(i)
                              for i in range(self.led_strip.num_leds)}

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
        self._current_led = i

    @current_x.setter
    def current_x(self, x: int):
        self._current_x = x

    @current_y.setter
    def current_y(self, y: int):
        self._current_y = y

    @current_frame.setter
    def current_frame(self, frame_idx: int):
        self._current_frame = frame_idx

    def set_ui_settings(self):
        """To override. Force general settings from village.settings
        such as area ROIs, White or Black for detection, etc.
        Check settings.py for what settings are available.
        Example:
            Set AREA4_BOX ROI:
                                           L,   T,   R,   B,   Thr
                settings.set("AREA4_BOX", [100, 110, 120, 200, 50])
            Set AREA4_BOX usage:
                settings.set("USAGE4_BOX", use) --- where
                USE in "ALLOWED" | "NOT_ALLOWED" | "TRIGGER" | "OFF"
        """
        pass

    def softcode_callback(self):
        """To override if needed. Called from a
        softcode to do something in Task"""
        pass

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
            spec = importlib.util.spec_from_file_location(module_name,
                                                          functions_path)
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

    def save_led_calibration(self):
        for i in range(self.led_strip.num_leds):
            self.led_positions[i].finalize()

        out = {int(k): {"x": int(v.x_hat) if v.x_hat is not None else -1,
                        "y": int(v.y_hat) if v.y_hat is not None else -1}
               for k, v in self.led_positions.items()}

        path = Path(settings.get("DATA_DIRECTORY"),
                    self.CALIBRATION_PATH + ".json")
        path.write_text(json.dumps(out, indent=2))
        self.register_value("led_strip_calibration_path", str(path))
        self.register_value(self.CALIBRATION_PATH + ".json", out)

    def load_led_calibration(self):
        path = Path(settings.get("DATA_DIRECTORY"),
                    self.CALIBRATION_PATH + ".json")
        with open(path) as f:
            calibration = json.load(f)

        for k, v in calibration.items():
            self.led_positions[int(k)] = LEDPosition(idx=int(k),
                                                     x_hat=v['x'],
                                                     y_hat=v['y'])
        error_str = f"Mismatch num_leds between calibration and led strip:" \
                    f" {len(self.led_positions)} vs {self.led_strip.num_leds}"
        assert len(self.led_positions) == self.led_strip.num_leds, error_str

    def _close_strip(self):
        self.accept_frames.clear()
        self.led_strip.clear_strip()
        self.led_strip.update_strip()

    def start(self):
        # A start() that all subclasses of TowersTask will do
        self.reload_softcode()

    def after_trial(self):
        pass

    def close(self):
        pass
