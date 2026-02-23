from village.manager import manager

task = manager.task
N_LEDS = 10
if hasattr(task.led_strip, 'num_leds'):
    N_LEDS = task.led_strip.num_leds
SOFTCODE_LED_OFF = 60
SOFTCODE_LED_BASE = 40

assert SOFTCODE_LED_BASE + N_LEDS <= SOFTCODE_LED_OFF, "Not enough softcodes for all LEDs"


def _set_led_color(i: int, red: int = 255, green: int = 255, blue: int = 255,
                   verbose: bool = True):
    task.led_strip.set_led_color(i, red, green, blue)
    task.led_strip.update_strip()
    task.current_led = i
    if verbose:
        print(f"[LED] ON index={i}")

def _clear_strip(verbose: bool = True):
    task.led_strip.clear_strip()
    task.led_strip.update_strip()
    if verbose:
        print("[LEDS] OFF")

def function11():
    print("Now accepting")
    task.accept_frames.set()

def function12():
    print("Now refusing")
    task.accept_frames.clear()

def function13():
    if not hasattr(task, 'accept_frames'):
        # Throws error because accept_frames not is Base Task class? We reload softcodes anyway when creating the Task
        return
    if task.accept_frames.is_set():
        task.current_x = task.cam_box.x_position
        task.current_y = task.cam_box.y_position
        task.current_frame = task.cam_box.frame_number
        task.add_sample()

def function60():
    _clear_strip()

for i in range(N_LEDS):
    def build_softcode(i=i):
        def func(i=i):
            _set_led_color(i)
        func.__name__ = f"function{SOFTCODE_LED_BASE + i}"
        return func
    globals()[f"function{SOFTCODE_LED_BASE + i}"] = build_softcode(i)
