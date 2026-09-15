"""TowersTask plus the online speed estimator"""

import time

from online_speed_estimator import OnlineSpeedEstimator, px_to_cm_fit
from tower_task import TowersTask
from village.scripts.log import log


class SlowTowersTask(TowersTask):
    """TowersTask, instrumented with the online corridor-speed estimator."""

    def __init__(self):
        super().__init__()
        self.speed_estimator = OnlineSpeedEstimator()
        self._to_cm = None
        self._base_adv: list = []
        self._reset_speed_stats()

    def _reset_speed_stats(self) -> None:
        self._n_frames = 0
        self._n_estimated = 0
        self._n_fast = 0
        self._flips = 0
        self._last_state = None
        self._max_speed = 0.0
        self._cost_s = 0.0

    def start(self):
        super().start()
        try:
            self._to_cm = px_to_cm_fit(self.led_positions)
        except Exception:
            self._to_cm = None
            log.error("[speed test] px->cm fit failed, estimator disabled",
                      exception=True)
            return
        self.speed_estimator.reset()
        self._reset_speed_stats()
        s = self.speed_estimator
        log.info(f"[speed test] armed: th_hi={s.th_hi} th_lo={s.th_lo} "
                 f"dwell={s.dwell} window={s.window} ends={s.ends}")

    def _update_hud(self) -> None:
        """Append the speed HUD row to the existing HUD items."""
        super()._update_hud()
        hud = self.cam_box.items_to_draw.get("hud")
        self._base_adv = list(hud.get("adv_label", [])) if hud else []

    def softcode_callback(self):
        """Run the task unchanged, then feed the estimator."""
        super().softcode_callback()
        if self._to_cm is None:
            return

        t0 = time.perf_counter()
        x = self.current_x

        pos_cm = None if x is None or x < 0 else self._to_cm(x)
        t = getattr(self.cam_box, "camera_timestamp", None) or time.time()
        try:
            fast = self.speed_estimator.update(t, pos_cm)
        except Exception:
            log.error("[speed test] estimator update failed",
                      exception=True)
            self._to_cm = None
            return
        self._cost_s += time.perf_counter() - t0

        self._n_frames += 1
        speed = self.speed_estimator.speed
        if speed is not None:
            self._n_estimated += 1
            self._max_speed = max(self._max_speed, speed)
        self._n_fast += int(fast)
        if self._last_state is not None and fast != self._last_state:
            self._flips += 1
        self._last_state = fast

        self._draw_speed(speed, fast)

    def _draw_speed(self, speed, fast: bool) -> None:
        """One extra HUD row: current speed, green while the gate is open."""
        hud = self.cam_box.items_to_draw.get("hud")
        if hud is None:
            return
        value = "   --" if speed is None else f" {speed:5.1f}"
        hud["adv_label"] = self._base_adv + [
            ("Speed:", f"{value} cm/s", not fast)]

    def after_trial(self):
        super().after_trial()
        n = max(self._n_frames, 1)
        self.register_value("speed_frames", self._n_frames)
        self.register_value("speed_estimated_frac",
                            round(self._n_estimated / n, 3))
        self.register_value("speed_fast_frac", round(self._n_fast / n, 3))
        self.register_value("speed_flips", self._flips)
        self.register_value("speed_max_cm_s", round(self._max_speed, 1))
        self.register_value("speed_cost_us",
                            round(self._cost_s / n * 1e6, 1))
        log.info(f"[speed test] frames={self._n_frames} "
                 f"est={self._n_estimated / n:.2f} "
                 f"fast={self._n_fast / n:.2f} flips={self._flips} "
                 f"max={self._max_speed:.1f}cm/s "
                 f"cost={self._cost_s / n * 1e6:.1f}us")
        self._reset_speed_stats()
