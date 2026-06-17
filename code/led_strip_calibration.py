from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from village.custom_classes.task_base import BpodEvent as Event
from village.settings import settings
from tower_task_base import TowersTaskBase, Color


class LedStripCalibration(TowersTaskBase):
    """Sequentially turn ON each LED, capture its (x, y) with cam_box, turn it
    OFF, then repair / interpolate / diagnose and save the led_index -> (x, y)
    map from the camera's point of view.
    """

    Y_DRIFT_WARN = 5  # max acceptable vertical spread within a segment (px)
    SPACING_RATIO_WARN = 7  # gap (median above this probably a missing LED?)
    MAX_REPAIRS_PER_SEGMENT = 2  # fix segment if below this many missing LEDs
    DO_REPAIR = True  # patch isolated astray LEDs before saving
    DO_REFINE = True  # snap LEDs onto a per-segment fit (interpolation)
    FIT_DEGREE = 1  # 1=straight line + uniform spacing; 2-3=gentle curvature

    def __init__(self):
        super().__init__()
        self.on_time = 0.4  # LED ON duration (s) so camera can detect
        self.off_time = 0.05  # duration (s) between LEDs
        self.settle_time = 0.05  # duration (s) between LED ON and frame cap
        self.results: dict[int, tuple] = {}
        self.raw_results: dict[int, tuple] | None = None
        self.num_leds = 0

    def set_ui_settings(self):
        settings.set("AREA1_BOX", [55, 215, 585, 275, 120])  # L, T, R, B, Thr
        settings.set("USAGE1_BOX", "ALLOWED")
        settings.set("USAGE2_BOX", "OFF")
        settings.set("USAGE3_BOX", "OFF")
        settings.set("USAGE4_BOX", "OFF")
        settings.set("DETECTION_COLOR", "WHITE")

    def start(self):
        super().start()
        self.COLOR_ON = Color(1, 1, 1)
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
        self.num_leds = self.led_strip.num_leds
        for i in range(self.num_leds):
            self.led_positions[i].finalize()
        self.results = {
            i: (self.led_positions[i].x_hat, self.led_positions[i].y_hat)
            for i in range(self.num_leds)
        }
        self.raw_results = None

        if self.DO_REPAIR:
            self.repair_outliers()
        self.run_diagnostics()
        if self.DO_REFINE:
            self.refine_positions()

        self.save_calibration()
        self.make_plot()

    def repair_outliers(self) -> list[str]:
        """Try to patch LEDs that are out of place, as the LEDs are physically
        uniform and should be evenly linearly spaced. If a single LED is far
        from its neighbours, replace with the average of its two neighbours."""
        if not self.results:
            return []
        half = self.num_leds // 2
        msgs: list[str] = []
        for name, indices in (("RIGHT", range(0, half)),
                              ("LEFT", range(half, self.num_leds))):
            msgs.extend(self._repair_segment(name, list(indices)))
        return msgs

    def _repair_segment(self, name: str, indices: list[int]) -> list[str]:
        msgs: list[str] = []
        present = sorted(i for i in indices if self.results[i][0] is not None)
        if len(present) < 5:  # ~too few
            return msgs

        ks = np.array(present, dtype=float)
        xs = np.array([self.results[i][0] for i in present], dtype=float)
        ys = np.array([self.results[i][1] for i in present], dtype=float)

        pred_x = self._robust_line(ks, xs)
        pred_y = self._robust_line(ks, ys)
        dev = np.hypot(xs - pred_x, ys - pred_y)

        med = float(np.median(dev))
        mad = float(np.median(np.abs(dev - med))) or 1.0
        _stdev = med + 5.0 * 1.4826 * mad  # robust estimate of stddev
        thresh = max(self.Y_DRIFT_WARN, _stdev)

        present_set = set(present)
        outliers = {present[j] for j in range(len(present)) if dev[j] > thresh}
        if not outliers:
            return msgs
        if len(outliers) > self.MAX_REPAIRS_PER_SEGMENT:
            msg = (f"[LED strip calib] {name}: too many ({len(outliers)})"
                   f" lost LEDs, check detection settings and run again.")
            print(msg)
            return msgs + [msg]

        for j, i in enumerate(present):
            # only patch an isolated outlier when neighbours are fine
            if i not in outliers or (i - 1) in outliers or (i + 1) in outliers:
                continue
            if (i - 1) in present_set and (i + 1) in present_set:
                a, b = self.results[i - 1], self.results[i + 1]
                fix = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            else:  # endpoint: fallback to robust fit
                fix = (float(pred_x[j]), float(pred_y[j]))
            old = tuple(self.results[i])
            self.results[i] = (int(round(fix[0])), int(round(fix[1])))
            msg = (f"[LED strip calib] {name}: LED {i} was lost ({dev[j]}px), "
                   f"but now is found: {old} -> ({fix[0]:.0f}, {fix[1]:.0f})")
            print(msg)
            msgs.append(msg)
        return msgs

    @staticmethod
    def _robust_line(k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict v at indices k from a robust line fit (median pairwise
        slope, median intercept), insensitive to a few outliers."""
        dk = np.diff(k)
        slope = float(np.median(np.diff(v) / dk)) if np.all(dk != 0) else 0.0
        intercept = float(np.median(v - slope * k))
        return slope * k + intercept

    def refine_positions(self) -> list[str]:
        """Snap every LED onto a per-segment fit of position vs index."""
        if not self.results:
            return []
        if self.raw_results is None:
            self.raw_results = dict(self.results)
        half = self.num_leds // 2
        msgs: list[str] = []
        for name, indices in (("RIGHT", range(0, half)),
                              ("LEFT", range(half, self.num_leds))):
            msgs.extend(self._refine_segment(name, list(indices)))
        return msgs

    def _refine_segment(self, name: str, indices: list[int]) -> list[str]:
        present = sorted(i
                         for i in indices
                         if self.raw_results[i][0] is not None)
        if len(present) < self.FIT_DEGREE + 1:
            msg = (f"[LED strip calib] {name}: too few LEDs ({len(present)})"
                   f" to fit deg {self.FIT_DEGREE}")
            print(msg)
            return [msg]

        ks = np.array(present, dtype=float)
        xs = np.array([self.raw_results[i][0] for i in present], dtype=float)
        ys = np.array([self.raw_results[i][1] for i in present], dtype=float)
        cx = np.polyfit(ks, xs, self.FIT_DEGREE)
        cy = np.polyfit(ks, ys, self.FIT_DEGREE)

        all_k = np.array(indices, dtype=float)
        fx, fy = np.polyval(cx, all_k), np.polyval(cy, all_k)
        for pos, idx in enumerate(indices):
            self.results[idx] = (int(round(fx[pos])), int(round(fy[pos])))

        res = np.hypot(xs - np.polyval(cx, ks), ys - np.polyval(cy, ks))
        spacing = np.abs(np.diff(fx))
        msg = (f"[LED strip calib] {name}: fit deg {self.FIT_DEGREE}, "
               f"moved raw points by mean={res.mean():.2f}px "
               f"max={res.max():.2f}px; "
               f"fitted spacing mean={spacing.mean():.2f}px "
               f"({len(indices)} LEDs, {len(present)} detected)")
        print(msg)
        return [msg]

    def run_diagnostics(self) -> tuple[bool, list[str]]:
        if not self.results:
            return False, ["No results to check (run the calibration first)."]

        half = self.num_leds // 2
        segments = [("RIGHT", list(range(0, half))),
                    ("LEFT", list(range(half, self.num_leds)))]
        all_ok = True
        lines: list[str] = []
        for seg_name, indices in segments:
            ok, seg_lines = self._diagnose_segment(seg_name, indices)
            all_ok = all_ok and ok
            lines.extend(seg_lines)

        if all_ok:
            print("LED calibration diagnostics: all segments OK")
        else:
            print("LED calibration diagnostics: issues found (see above)")
        return all_ok, lines

    def _diagnose_segment(self, name: str,
                          indices: list[int]) -> tuple[bool, list[str]]:
        detected = [(i, *self.results[i])
                    for i in indices if self.results[i][0] is not None]
        missing = [i for i in indices if self.results[i][0] is None]
        lines: list[str] = []
        ok = not missing

        if missing:
            lines.append(f"[LED strip calib] {name}:"
                         f" {len(missing)} LED(s) never detected: {missing}")
            print(f"[LED strip calib] {name}:"
                  f" {len(missing)} LED(s) never detected: {missing}")
        if len(detected) < 2:
            msg = (f"[LED strip calib] {name}:"
                   f" too few detected LEDs ({len(detected)}) to diagnose")
            print(msg)
            return False, lines + [msg]

        xs = np.array([d[1] for d in detected], dtype=float)
        ys = np.array([d[2] for d in detected], dtype=float)
        dx = np.diff(xs)

        direction = np.sign(np.median(dx)) or 1.0
        inversions = [(detected[k][0], detected[k + 1][0])
                      for k in range(len(dx))
                      if np.sign(dx[k]) not in (direction, 0.0)]
        dir_str = "increasing" if direction > 0 else "decreasing"
        if inversions:
            ok = False
            msg = (f"[LED strip calib] {name}: X not monotonic ({dir_str}):"
                   f" {len(inversions)} inversion(s) {inversions}")
            print(msg)
            lines.append(msg)
        else:
            lines.append(f"[LED strip calib] {name}: X monotonic ({dir_str}),"
                         f" {len(detected)} LEDs - OK")

        # Y
        y_spread = float(ys.max() - ys.min())
        if y_spread > self.Y_DRIFT_WARN:
            ok = False
            msg = (f"[LED strip calib] {name}: Y drift {y_spread:.1f}px >"
                   f" {self.Y_DRIFT_WARN}px (std={ys.std():.1f})")
            print(msg)
            lines.append(msg)
        else:
            lines.append(f"[LED strip calib] {name}: "
                         f"Y drift {y_spread:.1f}px (std={ys.std():.1f}) - OK")

        # X
        spacing = np.abs(dx)
        median_sp = float(np.median(spacing))
        lines.append(f"[LED strip calib] {name}: spacing px: "
                     f"mean={spacing.mean():.1f} median={median_sp:.1f} "
                     f"min={spacing.min():.1f} max={spacing.max():.1f}")
        print(lines[-1])
        if median_sp > 0:
            big = [(detected[k][0], detected[k + 1][0], round(spacing[k], 1))
                   for k in range(len(spacing))
                   if spacing[k] > self.SPACING_RATIO_WARN * median_sp]
            if big:
                ok = False
                msg = f"[LED strip calib] {name}: possible missing LED: {big}"
                print(msg)
                lines.append(msg)

        return ok, lines

    def save_calibration(self):
        """Write led_index -> (x, y) to JSON (same format/path so main Task
        reads back with load_led_calibration). Uses the refined results
        directly, so it does not re-finalize from samples."""
        out = {
            int(i): {
                "x": int(x) if x is not None else -1,
                "y": int(y) if y is not None else -1,
            }
            for i, (x, y) in self.results.items()
        }
        path = Path(settings.get("DATA_DIRECTORY"),
                    self.CALIBRATION_PATH + ".json")
        path.write_text(json.dumps(out, indent=2))
        self.register_value("led_strip_calibration_path", str(path))
        self.register_value(self.CALIBRATION_PATH + ".json", out)

    def make_plot(self):
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        half = self.num_leds // 2
        if self.raw_results:
            rxs = [v[0] for v in self.raw_results.values() if v[0] is not None]
            rys = [v[1] for v in self.raw_results.values() if v[1] is not None]
            ax.scatter(rxs, rys, s=18, facecolors="none", edgecolors="0.6",
                       linewidths=0.5, label="raw")
        for indices, color, label in ((range(0, half), "tab:blue", "right"),
                                      (range(half, self.num_leds),
                                       "tab:orange", "left")):
            xs = [self.results[i][0] for i in indices
                  if self.results[i][0] is not None]
            ys = [self.results[i][1] for i in indices
                  if self.results[i][1] is not None]
            ax.scatter(xs, ys, s=12, color=color, label=label)
        ax.set_xlim(0, 640)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.legend()
        path = Path(settings.get("DATA_DIRECTORY"),
                    self.CALIBRATION_PATH + ".png")
        plt.savefig(path)
