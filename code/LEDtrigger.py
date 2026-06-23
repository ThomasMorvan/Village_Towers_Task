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
            self.task.execute_function(3)
        except Exception:
            print("Error running function" + str(3))

        # Stage 0:
        #    Step 1: deliver reward when mouse gets in ROI instead of poke
        #    Step 2: now requires a real poke.
        try:
            if (self.task._odc.stage == 0
                    and getattr(self.task.settings,
                                "proximity_trigger", True)):
                sma = self.task.bpod.sma
                # current_state is NaN (float) once the trial's SMA finishes
                if not isinstance(sma.current_state, int):
                    print("[LEDTrigger] wrong state type:", sma.current_state,
                          type(sma.current_state))
                    return
                state = sma.state_names[sma.current_state]
                if state == "WAIT FOR CHOICE POKE":
                    if cam.area2_is_triggered:
                        self.task.bpod.poke(1)  # left port
                    elif cam.area3_is_triggered:
                        self.task.bpod.poke(3)  # right port
                elif state == "POKE MIDDLE":
                    if cam.area4_is_triggered:
                        self.task.bpod.poke(2)  # center port
        except (IndexError, AttributeError) as e:
            print(f"[LEDTrigger] Error processing LED trigger: {e}")
            pass
