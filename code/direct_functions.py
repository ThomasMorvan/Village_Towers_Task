import time
import threading
from contextlib import nullcontext
from village.custom_classes.direct_functions_base import DirectFunctionsBase


class DirectFunctions(DirectFunctionsBase):
    def function10(self):
        """Turn off all LEDs (SOFTCODE_LED_OFF)."""
        self._clear_strip()

    def function1(self):
        """Accept frames from camera for LED position capture."""
        self.task.accept_frames.set()
        print("Now accepting frames")

    def function2(self):
        """Refuse frames from camera for LED position capture."""
        self.task.accept_frames.clear()
        print("Now refusing frames")

    def function3(self):
        """Softcode function to be called at every frame processed.
        If we accept frames, update current (x, y) position then call
        softcode_callback (user defined function that does stuff
        in response to the new position)."""
        if not hasattr(self.task, 'accept_frames'):
            # Throws error because accept_frames not in Base Task class?
            # We reload softcodes anyway when creating the Task
            return
        if self.task.accept_frames.is_set():
            self.task.current_x = self.task.cam_box.x_position
            self.task.current_y = self.task.cam_box.y_position
            self.task.current_frame = self.task.cam_box.frame_number
            self.task.softcode_callback()

    def function4(self):
        """Softcode function to turn ON current LED, wait a bit,
        accept frames, wait for capture a few frames, reject frames,
        then turn OFF the LED."""
        if not hasattr(self.task, "led_strip"):
            # Throws error because led_strip not in Base Task class?
            # We reload softcodes anyway when creating the Task
            return
        self._set_led_color(self.task.current_led, *self.task.COLOR_ON)
        time.sleep(self.task.settle_time)

        self.task.accept_frames.set()
        time.sleep(self.task.on_time)
        self.task.accept_frames.clear()

        self._set_led_color(self.task.current_led, *self.task.COLOR_OFF)

    def function5(self):
        """Turn ON current_led, then OFF after led_on_duration. Same as the
        old on/sleep/off, but a Timer replaces the sleep so it doesn't block
        the camera loop. The timer is tracked on the task so it can be
        cancelled at trial end (an async off can outlive the trial; the old
        synchronous sleep always finished before the trial ended)."""
        if not hasattr(self.task, "led_strip"):
            # Throws error because led_strip not in Base Task class?
            # We reload softcodes anyway when creating the Task
            return
        leds = self.task.current_led
        self._set_led_color(leds, *self.task.COLOR_ON)
        timer = threading.Timer(
            self.task.led_on_duration, self._set_led_color,
            args=(leds, *self.task.COLOR_OFF), kwargs={"verbose": False})
        self.task._led_timers.append(timer)
        timer.start()

    def function6(self):
        """Turn ON all LEDs in current_led without a
        timeout (always-on stages)."""
        if not hasattr(self.task, "led_strip"):
            return
        self._set_led_color(self.task.current_led, *self.task.COLOR_ON)

    def function7(self):
        print("Port 1 poke")

    def function8(self):
        print("Port 2 poke")

    def function9(self):
        print("Port 3 poke")

    def _set_led_color(self, i: int | list[int],
                       red: int = 255, green: int = 255, blue: int = 255,
                       verbose: bool = True):
        """Set LED color for one index or a list of indices,
        and update the strip only once."""
        indices = i if isinstance(i, list) else [i]
        with getattr(self.task, "led_lock", None) or nullcontext():
            for idx in indices:
                self.task.led_strip.set_led_color(idx, red, green, blue)
            self.task.led_strip.update_strip(sleep_duration=None)
        if verbose:
            print(f"[LED] {i} -> [{red} {green} {blue}]")

    def _clear_strip(self):
        """Turn off all LEDs."""
        with getattr(self.task, "led_lock", None) or nullcontext():
            self.task.led_strip.clear_strip()
            self.task.led_strip.update_strip(sleep_duration=None)
