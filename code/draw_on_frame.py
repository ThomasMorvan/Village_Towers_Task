import cv2
from village.classes.abstract_classes import CameraBase
from village.custom_classes.camera_draw_base import CameraDrawBase


class DrawFurthestX(CameraDrawBase):
    def __init__(self) -> None:
        self.name = "Draw Furthest X"
        self.furthest_x = -1
        self.next_trigger = 0
        self.led_pos = []
        self.animal_trace = []

    def draw(self, cam: CameraBase) -> None:
        self.furthest_x = cam.items_to_draw.get("furthest_x", -1)
        self.next_trigger = cam.items_to_draw.get("next_trigger", -1)
        self.led_pos = cam.items_to_draw.get("led_pos", -1)

        line_lims = [0, cam.height]
        try:
            area_1 = cam.areas[0]
            L, T, R, B = area_1[0], area_1[1], area_1[2], area_1[3]
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
