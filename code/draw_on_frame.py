from __future__ import annotations
from math import radians, cos, sin
import cv2
import numpy as np
from typing import TYPE_CHECKING
from village.custom_classes.camera_draw_base import CameraDrawBase
from task_stages import PHASE_BGR, Difficulty

if TYPE_CHECKING:
    from village.devices.camera import Camera

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_size(text: str, scale: float,
               thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h


class DrawFurthestX(CameraDrawBase):
    def __init__(self) -> None:
        self.name = "Draw Furthest X"
        self.furthest_x = -1
        self.next_trigger = 0
        self.led_pos = []
        self.animal_trace = []

    def _draw_hud(self, frame: np.ndarray, hud: dict) -> None:
        MIN_X = 320
        ROW_1_Y = 15
        HUD_HEIGHT = 40
        HUD_COLOR = (75, 75, 75)
        GOOD_COLOR = (40, 180, 40)
        BAD_COLOR = (40, 40, 220)
        TEXT_SIZE = 0.4
        BIG_TEXT_SIZE = 0.5
        SMALL_TEXT_SIZE = 0.3
        phase = hud["phase"]

        # background fill
        cv2.rectangle(frame, (MIN_X, 0), (frame.shape[1], HUD_HEIGHT),
                      HUD_COLOR, -1)

        # current phase
        curr_x = MIN_X + 4
        badge_color = PHASE_BGR.get(phase, (80, 80, 80))
        badge_label = phase.upper()
        phase_badge_w = _text_size(badge_label, BIG_TEXT_SIZE)[0] + 12
        cv2.rectangle(frame, (curr_x, 0),
                      (curr_x + phase_badge_w, HUD_HEIGHT - 20),
                      badge_color, -1)
        cv2.putText(frame, badge_label,
                    (MIN_X + 10, (HUD_HEIGHT - 20) // 2 + 5),
                    _FONT, BIG_TEXT_SIZE, (255, 255, 255), 1, cv2.LINE_AA)

        # current stage
        if phase == "warmup":
            warmup_trial, n_warmup_trials = hud.get("warmup_trial", (0, 1))
            stage_label = f"{warmup_trial}/{n_warmup_trials}"
        else:
            stage_label = f"S{hud['stage']}: {hud['stage_name']}"
        cv2.putText(frame, stage_label, (curr_x, HUD_HEIGHT - 5),
                    _FONT, TEXT_SIZE, (210, 210, 210), 1, cv2.LINE_AA)
        curr_x += 98 + 7  # max stage size is BackForth at 98

        cv2.line(frame, (curr_x, 8), (curr_x, HUD_HEIGHT - 8),
                 (200, 200, 200), 1)
        curr_x += 7

        # Difficulty
        diff = hud.get("difficulty", Difficulty())
        ck = hud.get("checkpoint", 0)
        diff_labels = [f"mu_r: {diff.mu_r:.2f}",
                       f"mu_nr: {diff.mu_nr:.2f}",
                       f"led ms: {diff.led_ms:.0f}",
                       f"Checkpoint: {ck}"]
        diff_y = 8
        for label in diff_labels:
            cv2.putText(frame, label, (curr_x, diff_y),
                        _FONT, SMALL_TEXT_SIZE,
                        (200, 200, 200), 1, cv2.LINE_AA)
            # print(_text_size(label, SMALL_TEXT_SIZE))
            diff_y += 10

        curr_x += 66 + 7  # max label is checkpoint at 87
        cv2.line(frame, (curr_x, 8), (curr_x, HUD_HEIGHT - 8),
                 (200, 200, 200), 1)
        curr_x += 7

        # current accuracy
        accuracy = hud.get("rolling_acc")
        if accuracy is not None:
            perc = int(100 * accuracy)
            r = max(0, int(255 * (1 - accuracy)))
            g = max(0, int(255 * accuracy))
            acc_label = f"Acc {perc}%"
            cv2.putText(frame, acc_label, (curr_x, ROW_1_Y),
                        _FONT, TEXT_SIZE, (r, g, 60), 1, cv2.LINE_AA)
            # print(_text_size(acc_label, TEXT_SIZE))

        # streak
        streak = hud.get("streak", 0)
        streak_color = GOOD_COLOR if streak >= 0 else BAD_COLOR
        streak_label = f"Streak {streak:+d}"
        cv2.putText(frame, streak_label, (curr_x, 33),
                    _FONT, TEXT_SIZE, streak_color, 1, cv2.LINE_AA)
        curr_x += 70

        cv2.line(frame, (curr_x, 8), (curr_x, HUD_HEIGHT - 8),
                 (200, 200, 200), 1)
        curr_x += 5

        # Next stage criterion
        adv = hud.get("adv_label")
        if adv:
            y = 8
            for i, line in enumerate(adv if isinstance(adv, list) else [adv]):
                txt1, txt2, tf = line
                color = GOOD_COLOR if tf else (160, 160, 160)
                cv2.putText(frame, txt1, (curr_x, y),
                            _FONT, SMALL_TEXT_SIZE,
                            color, 1, cv2.LINE_AA)
                cv2.putText(frame, txt2, (curr_x, y + 10),
                            _FONT, SMALL_TEXT_SIZE,
                            color, 1, cv2.LINE_AA)
                y += 20

    def _draw_accumulator(self, frame: np.ndarray, anm) -> None:
        """draw proba + distribution of choice from DDM"""
        acc = anm.acc
        FRAME_WIDTH = frame.shape[1]
        UPPER_BOX_Y = 40
        PROBA = acc.P.copy()
        xc = acc.xc
        bias = acc._bias
        p_right = acc.p_right()

        # widget size and pos
        radius_out, radius_in = 32, 20
        DISTRIBUTION_H = 11  # distrib H above the circle
        WIDGET_WIDTH = radius_out * 2 + 16
        WIDGET_HEIGHT = radius_out + DISTRIBUTION_H + 32
        WIDGET_X = FRAME_WIDTH - WIDGET_WIDTH - 6
        WIDGET_Y = UPPER_BOX_Y + 3
        cx = WIDGET_X + WIDGET_WIDTH // 2
        cy = WIDGET_Y + DISTRIBUTION_H + radius_out + 4

        # Background box
        cv2.rectangle(frame, (WIDGET_X, WIDGET_Y),
                      (WIDGET_X + WIDGET_WIDTH, WIDGET_Y + WIDGET_HEIGHT),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (WIDGET_X, WIDGET_Y),
                      (WIDGET_X + WIDGET_WIDTH, WIDGET_Y + WIDGET_HEIGHT),
                      (60, 60, 60), 1)

        # Distribution bars outward from the arc
        x_min = float(xc[0])
        x_span = float(xc[-1]) - x_min or 1.0
        max_p = float(PROBA.max()) or 1.0
        for (p_i, x_i) in zip(PROBA, xc):
            frac = (x_i - x_min) / x_span
            angle = radians(180 + frac * 180)
            ca, sa = cos(angle), sin(angle)
            bar_h = max(1, int(p_i / max_p * DISTRIBUTION_H))
            x0 = int(cx + radius_out * ca)
            y0 = int(cy + radius_out * sa)
            x1 = int(cx + (radius_out + bar_h) * ca)
            y1 = int(cy + (radius_out + bar_h) * sa)
            if x_i >= bias:
                col = (0, 200, 60)
            else:
                col = (0, 60, 220)
            cv2.line(frame, (x0, y0), (x1, y1), col, 2)

        # arc
        cv2.ellipse(frame, (cx, cy), (radius_out, radius_out),
                    0, 180, 360, (65, 65, 65), 3)

        # p_right arrow
        angle = radians(180 + p_right * 180)
        ca, sa = cos(angle), sin(angle)
        end = (int(cx + radius_out * ca), int(cy + radius_out * sa))
        start = (int(cx - 6 * ca), int(cy - 6 * sa))
        cv2.line(frame, start, end, (255, 140, 0), 2)
        cv2.circle(frame, (cx, cy), radius_in, (22, 22, 22), -1)  # to hide

        # p
        col1 = int((1.0 - p_right) * 200)
        col2 = int(p_right * 200)
        val = f"{p_right:.2f}"
        cv2.putText(frame, val, (cx - _text_size(val, 0.42)[0] // 2, cy - 2),
                    _FONT, 0.42, (40, col2, col1), 1, cv2.LINE_AA)

    def draw(self, cam: Camera) -> None:
        self.furthest_x = cam.items_to_draw.get("furthest_x", -1)
        self.next_trigger = cam.items_to_draw.get("next_trigger", -1)
        self.led_pos = cam.items_to_draw.get("led_pos", -1)

        line_lims = [0, cam.height]
        try:
            area_1 = cam.areas[0]
            _, T, _, B = area_1[0], area_1[1], area_1[2], area_1[3]
            line_lims = [T, B]
        except Exception:
            pass

        if self.furthest_x != -1:
            cv2.line(cam.frame,
                     (self.furthest_x, line_lims[0]),
                     (self.furthest_x, line_lims[1]), (0, 255, 255), 2)

        if self.next_trigger != -1:
            cv2.line(cam.frame,
                     (self.next_trigger, line_lims[0]),
                     (self.next_trigger, line_lims[1]), (255, 0, 255), 2)

        if isinstance(self.led_pos, list) and self.led_pos:
            for pos in self.led_pos:
                cv2.circle(cam.frame, (pos.x_hat, pos.y_hat), 3,
                           (0, 255, 0), 2)

        maxlen = 25 * 5
        anm = cam.items_to_draw.get("auto_instance")

        if anm is not None:
            if anm.position is not None:
                cv2.circle(cam.frame, anm.position,
                           cam.detection_size, (255, 200, 0), -1)
            trace = list(anm.trace)
            n = len(trace)
            if n > 1:
                for i in range(1, n):
                    age = n - 1 - i
                    intensity = max(0, int(255 * (maxlen - age) / maxlen))
                    cv2.line(cam.frame, trace[i - 1], trace[i],
                             (0, intensity, intensity + 80), 2, cv2.LINE_AA)
        else:
            self.animal_trace = list(
                cam.items_to_draw.get("animal_trace", [])
            )
            n = len(self.animal_trace)
            if n > 1:
                for i in range(1, n):
                    age = n - 1 - i
                    intensity = max(0, int(255 * (maxlen - age) / maxlen))
                    cv2.line(cam.frame, self.animal_trace[i - 1],
                             self.animal_trace[i],
                             (0, intensity, intensity), 2, cv2.LINE_AA)

        hud = cam.items_to_draw.get("hud")
        if hud:
            self._draw_hud(cam.frame, hud)

        if anm is not None and hasattr(anm, "acc"):
            self._draw_accumulator(cam.frame, anm)
