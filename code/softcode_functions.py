import time
from village.manager import manager

task = manager.task

def _set_led_color(i: int, red: int = 255, green: int = 255, blue: int = 255,
                   verbose: bool = True):
    task.led_strip.set_led_color(i, red, green, blue)
    task.led_strip.update_strip(sleep_duration=None)
    if verbose:
        print(f"[LED] {i} -> [{red} {green} {blue}] for {task.led_on_duration}s")


def _clear_strip():
    task.led_strip.clear_strip()
    task.led_strip.update_strip(sleep_duration=None)

def function0():
    _clear_strip()

def function1():
    task.accept_frames.set()
    print("Now accepting frames")

def function2():
    task.accept_frames.clear()
    print("Now refusing frames")

def function3():
    if not hasattr(task, 'accept_frames'):
        # Throws error because accept_frames not is Base Task class? We reload softcodes anyway when creating the Task
        return
    if task.accept_frames.is_set():
        task.current_x = task.cam_box.x_position
        task.current_y = task.cam_box.y_position
        task.current_frame = task.cam_box.frame_number
        task.softcode_callback()

def function4():
    if not hasattr(task, "led_strip"):
        # Throws error because accept_frames not is Base Task class? We reload softcodes anyway when creating the Task
        return
    _set_led_color(task.current_led, *task.COLOR_ON)
    time.sleep(task.settle_time)

    task.accept_frames.set()
    time.sleep(task.on_time)
    task.accept_frames.clear()

    _set_led_color(task.current_led, *task.COLOR_OFF)

def function5():
    if not hasattr(task, "led_strip"):
        # Throws error because accept_frames not is Base Task class? We reload softcodes anyway when creating the Task
        return
    _set_led_color(task.current_led, *task.COLOR_ON)
    time.sleep(task.led_on_duration)
    _set_led_color(task.current_led, *task.COLOR_OFF, verbose=False)
