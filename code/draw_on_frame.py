import cv2
from village.classes.abstract_classes import CameraBase
from village.custom_classes.camera_draw_base import CameraDrawBase


class DrawFurthestX(CameraDrawBase):
    def __init__(self) -> None:
        """Initializes the DrawFurthestX instance."""
        self.name = "Draw Furthest X"
        self.furthest_x = -1
        self.next_trigger = 0
        self.led_pos = []

    def draw(self, cam: CameraBase) -> None:
        """Draws a vertical line at furthest_x on the frame.

        Args:
            camera_instance (CameraBase): The camera instance
            providing the frame to draw on.
        """
        self.furthest_x = cam.items_to_draw.get("furthest_x", -1)
        self.next_trigger = cam.items_to_draw.get("next_trigger", -1)
        self.led_pos = cam.items_to_draw.get("led_pos", -1)
        if self.furthest_x != -1:
            cv2.line(
                cam.frame,
                (self.furthest_x, 0),
                (self.furthest_x, cam.height),
                (0, 255, 255),
                2,
            )

        if self.next_trigger != -1:
            cv2.line(
                cam.frame,
                (self.next_trigger, 0),
                (self.next_trigger, cam.height),
                (255, 0, 255),
                2,
            )

        if isinstance(self.led_pos, list) and self.led_pos != []:
            for pos in self.led_pos:
                cv2.line(
                    cam.frame,
                    (pos, 0),
                    (pos, cam.height),
                    (0, 255, 0),
                    2,
                )
