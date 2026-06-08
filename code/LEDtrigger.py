from __future__ import annotations

from typing import TYPE_CHECKING
from village.custom_classes.camera_trigger_base import CameraTriggerBase
from village.manager import manager

if TYPE_CHECKING:
    from village.devices.camera import Camera


class LEDTrigger(CameraTriggerBase):
    def __init__(self) -> None:
        """Initializes the CameraTriggerBase instance."""
        self.name = "LED Trigger"

    def trigger(self, cam: Camera) -> None:
        """Called everytime a frame is processed. Here softcode13 updates
        detected (x, y) position in Task so we can react to it.

        Args:
            cam (Camera): The camera instance providing the trigger status.
        """

        try:
            self.functions[3]()
        except Exception:
            print("Error running function" + str(3))
