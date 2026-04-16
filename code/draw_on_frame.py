import cv2
from village.custom_classes.camera_draw_base import CameraDrawBase


class DrawFurthestX(CameraDrawBase):
    def __init__(self) -> None:
        """Initializes the DrawFurthestX instance."""
        self.name = "Draw Furthest X"
        self.furthest_x = -1
        self.next_trigger = 0


    def draw(self, camera_intance) -> None:
        """Draws a vertical line at furthest_x on the frame.

        Args:
            camera_intance (CameraBase): The camera instance providing the frame to draw on.
        """
        self.furthest_x = camera_intance.items_to_draw.get("furthest_x", -1)
        self.next_trigger = camera_intance.items_to_draw.get("next_trigger", -1)
        if self.furthest_x != -1:
            cv2.line(
                camera_intance.frame,
                (self.furthest_x, 0),
                (self.furthest_x, camera_intance.height),
                (0, 255, 255),
                2,
            )

        if self.next_trigger != -1:
            cv2.line(
                camera_intance.frame,
                (self.next_trigger, 0),
                (self.next_trigger, camera_intance.height),
                (255, 0, 255),
                2,
            )
