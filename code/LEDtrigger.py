from village.classes.abstract_classes import CameraBase
from village.custom_classes.camera_trigger_base import CameraTriggerBase
from village.manager import manager

# Overriding CameraTriggerBase to run a softcode at every frame processed.
# Here softcode updates detected (x, y) position in Task so we can react to it.


class LEDTrigger(CameraTriggerBase):
    def __init__(self) -> None:
        """Initializes the CameraTriggerBase instance."""
        self.name = "LED Trigger"

    def trigger(self, cam: CameraBase) -> None:
        """Called everytime a frame is processed. Here softcode13 updates
        detected (x, y) position in Task so we can react to it.

        Args:
            cam (CameraBase): The camera instance providing the trigger status.
        """

        manager.run_softcode_function(3)
