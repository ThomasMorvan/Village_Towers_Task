from __future__ import annotations

from collections import deque

from village.custom_classes.task_base import BpodEvent as Event
from village.settings import settings
from village.scripts.log import log
from tower_task_base import TowersTaskBase


class DistanceOffsetCalibration(TowersTaskBase):
    """task to calibrate distance_offset, the number of px IN FRONT of
    the animal centroid at which a LED fires.

    How to use
    ----------
    1. Edit DISTANCE_OFFSET below and run this task.
    2. Put the 3D-printed mouse on the corridor floor (with a 10 cm ruler next
       to it for px<->cm scale) and slide it slowly from the entry towards the
       far end (decreasing x).
    3. Every LED on the chosen side (SIDE) is armed. Each one lights up the
       instant the centroid crosses (led.x_hat + DISTANCE_OFFSET), using the
       exact same proximity logic as the real task (_furthest_x + led_triggers,
       see tower_task._softcode_callback_proximity).
    4. Watch where the printed mouse physically is when each LED fires. The LED
       should fire 10cm in front of the LED. If it fires too early
       (centroid still far from the LED) lower DISTANCE_OFFSET; too late, raise
       it. Use the ruler to read the gap in cm, then run again.
    5. Pull the mouse back out past the entry to auto re-arm all LEDs and sweep
       again (no need to restart the task).
    """

    DISTANCE_OFFSET = 50  # px in front of the centroid that triggers a LED
    SIDE = "RIGHT"  # which physical half to arm: "RIGHT" (0-71) or "LEFT"
    SWEEP_TIMEOUT_S = 120  # seconds before a trial times out and re-arms

    # Set True to skip proximity triggering and instead verify LED positions:
    # place the object in front of each LED and the overlay shows the nearest
    # calibrated LED + px distance from the detected centroid to that LED.
    VERIFY_POSITIONS = False
    # ----------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._res_x = settings.get("CAM_BOX_RESOLUTION")[0]
        self.distance_offset = self.DISTANCE_OFFSET
        self.available_leds_idx: set[int] = set()
        self.used_leds_idx: set[int] = set()
        self.led_triggers: list[tuple[float, int]] = []
        self.next_trigger = self._res_x + 1
        self._furthest_x = self._res_x + 1
        self._rearm_x = self._res_x + 1
        self.trigger_log: list[tuple[int, int, int, int]] = []
        self.animal_trace: deque = deque(maxlen=25 * 5)
        self._verify_positions: dict | None = None  # built in start()

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [55, 220, 585, 258, 65])  # L, T, R, B, Thr
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "BLACK")

    def start(self):
        super().start()
        self.load_led_calibration()
        self.settings.iti_time = 0
        self.maximum_number_of_trials = 9999
        # Pre-build the lookup table used by verify mode: idx -> (x_hat, y_hat)
        if self.VERIFY_POSITIONS:
            self._verify_positions = {
                i: (p.x_hat, p.y_hat)
                for i, p in self.led_positions.items()
                if p.x_hat is not None and p.x_hat >= 0
            }

    def _half_indices(self) -> range:
        half = self.led_strip.num_leds // 2
        if self.SIDE.upper() == "LEFT":
            return range(half, self.led_strip.num_leds)
        return range(0, half)

    def _arm_leds(self):
        """Arm every calibrated LED on the chosen side."""
        self.distance_offset = self.DISTANCE_OFFSET
        self.available_leds_idx = {i for i in self._half_indices()
                                   if self.led_positions[i].x_hat is not None
                                   and self.led_positions[i].x_hat >= 0}
        self.used_leds_idx = set()
        self._furthest_x = self._res_x + 1
        self.cam_box.items_to_draw.pop("verify_nearest", None)
        if not self.VERIFY_POSITIONS:
            self._build_led_triggers()
            self._rearm_x = self.next_trigger
        self._publish_led_pos()
        log.info(f"[offset calib] armed {len(self.available_leds_idx)} LEDs on"
                 f" {self.SIDE} side, distance_offset={self.distance_offset}")

    def _build_led_triggers(self):
        self.led_triggers = sorted(
            [(self.led_positions[i].x_hat + self.distance_offset, int(i))
             for i in self.available_leds_idx], reverse=True)
        self.next_trigger = self.led_triggers[0][0] if self.led_triggers else 0
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger

    def _publish_led_pos(self):
        self.cam_box.items_to_draw["led_pos"] = [
            self.led_positions[i] for i in self.available_leds_idx]
        self.cam_box.items_to_draw["led_pos_used"] = [
            self.led_positions[i] for i in self.used_leds_idx]

    def create_trial(self):
        self._arm_leds()
        self.bpod.add_state(
            state_name="SLIDE_MOUSE",
            state_timer=self.SWEEP_TIMEOUT_S,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[("SoftCode", self.SOFTCODE_CAMERA_ACCEPT)],
        )

    def softcode_callback(self):
        if self.current_x is None or self.current_y is None:
            return
        self.animal_trace.append((int(self.current_x), int(self.current_y)))
        self.cam_box.items_to_draw["animal_trace"] = self.animal_trace

        if self.VERIFY_POSITIONS:
            self._verify_nearest()
            return

        # All LEDs fired and the mouse was pulled back out -> re-arm in place.
        if (not self.led_triggers and self.used_leds_idx
                and self.current_x > self._rearm_x):
            self._arm_leds()
            return

        self._proximity_trigger()

    def _verify_nearest(self):
        """Find the calibrated LED with the closest x to the current centroid
        and publish it so draw_preview can show the gap."""
        if not self._verify_positions:
            return
        cx, cy = float(self.current_x), float(self.current_y)
        best_idx, best_dist = None, float("inf")
        for idx, (lx, ly) in self._verify_positions.items():
            d = abs(cx - lx)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            return
        lx, ly = self._verify_positions[best_idx]
        dx = int(cx) - int(lx)  # signed: positive = centroid to the right of LED
        info = {"centroid": (int(cx), int(cy)), "led_idx": best_idx,
                "led_xy": (int(lx), int(ly)), "dx_px": dx}
        self.cam_box.items_to_draw["verify_nearest"] = info
        log.debug(f"[verify] LED {best_idx} nearest: dx={dx:+d}px "
                  f"centroid=({int(cx)},{int(cy)}) "
                  f"led=({int(lx)},{int(ly)})")

    def _proximity_trigger(self):
        """Fire LEDs one by one as the centroid passes them
        (same as tower_task._softcode_callback_proximity)."""
        if not self.led_triggers:
            return
        if self.current_x >= self._furthest_x:
            return

        self._furthest_x = self.current_x
        self.cam_box.items_to_draw["furthest_x"] = self._furthest_x

        if self._furthest_x > self.next_trigger:
            return

        triggered: list[int] = []
        while (self.led_triggers
               and self._furthest_x <= self.led_triggers[0][0]):
            _trigger_x, led_idx = self.led_triggers.pop(0)
            triggered.append(led_idx)
            self.available_leds_idx.discard(led_idx)
            self.used_leds_idx.add(led_idx)
            led_x = int(self.led_positions[led_idx].x_hat)
            centroid_x = int(self._furthest_x)
            gap = centroid_x - led_x
            self.trigger_log.append((led_idx, centroid_x, led_x, gap))
            log.info(f"[offset calib] LED {led_idx} fired: "
                     f"centroid_x={centroid_x} led_x={led_x} "
                     f"gap={gap}px (offset={self.distance_offset})")

        if triggered:
            # ALL_LEDS_ON leaves the LEDs lit (no timeout) so the gap to the
            # physical mouse stays visible.
            self.current_led = triggered
            try:
                self.execute_function(self.SOFTCODE_ALL_LEDS_ON)
            except Exception:
                log.error("Error running function "
                          + str(self.SOFTCODE_ALL_LEDS_ON))
            self._publish_led_pos()

        self.next_trigger = (self.led_triggers[0][0]
                             if self.led_triggers else self._res_x + 1)
        self.cam_box.items_to_draw["next_trigger"] = self.next_trigger

    def after_trial(self):
        self.register_value("distance_offset", self.distance_offset)
        self.register_value("side", self.SIDE)
        self.register_value("n_fired", len(self.trigger_log))
        self.register_value("trigger_log", self.trigger_log)
        self.trigger_log = []
        self.animal_trace = deque(maxlen=25 * 5)
        self.cam_box.items_to_draw["animal_trace"] = self.animal_trace

    def close(self):
        self._close_strip()
        items = getattr(self.cam_box, "items_to_draw", None)
        if isinstance(items, dict):
            items.clear()
