import time
from village.manager import manager

task = manager.task


def _set_led_color(i: int, red: int = 255, green: int = 255, blue: int = 255,
                   verbose: bool = True):
    """Set LED color."""
    task.led_strip.set_led_color(i, red, green, blue)
    task.led_strip.update_strip(sleep_duration=None)
    if verbose:
        print(f"[LED] {i} -> [{red} {green} {blue}]")


def _clear_strip():
    """Turn off all LEDs."""
    task.led_strip.clear_strip()
    task.led_strip.update_strip(sleep_duration=None)


def function0():
    _clear_strip()


def function1():
    """Accept frames from camera for LED position capture."""
    task.accept_frames.set()
    print("Now accepting frames")


def function2():
    """Refuse frames from camera for LED position capture."""
    task.accept_frames.clear()
    print("Now refusing frames")


def function3():
    """Softcode function to be called at every frame processed.
    If we accept frames, update current (x, y) position then call
    softcode_callback (user defined function that does stuff
    in response to the new position)."""
    if not hasattr(task, 'accept_frames'):
        # Throws error because accept_frames not in Base Task class?
        # We reload softcodes anyway when creating the Task
        return
    if task.accept_frames.is_set():
        task.current_x = task.cam_box.x_position
        task.current_y = task.cam_box.y_position
        task.current_frame = task.cam_box.frame_number
        task.softcode_callback()


def function4():
    """Softcode function to turn ON current LED, wait a bit,
    accept frames, wait for capture a few frames, reject frames,
    then turn OFF the LED."""
    if not hasattr(task, "led_strip"):
        # Throws error because led_strip not in Base Task class?
        # We reload softcodes anyway when creating the Task
        return
    _set_led_color(task.current_led, *task.COLOR_ON)
    time.sleep(task.settle_time)

    task.accept_frames.set()
    time.sleep(task.on_time)
    task.accept_frames.clear()

    _set_led_color(task.current_led, *task.COLOR_OFF)


def function5():
    """Softcode function to turn ON current LED, wait a bit,
    then turn OFF the LED. No frame capture."""
    if not hasattr(task, "led_strip"):
        # Throws error because led_strip not in Base Task class?
        # We reload softcodes anyway when creating the Task
        return
    _set_led_color(task.current_led, *task.COLOR_ON)
    time.sleep(task.led_on_duration)
    _set_led_color(task.current_led, *task.COLOR_OFF, verbose=False)


def function6():
    print("Port 1 poke")


def function7():
    print("Port 2 poke")


def function8():
    print("Port 3 poke")
