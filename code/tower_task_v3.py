"""TowersTask V3: a fixed number of cues per trial, split between the sides."""

from fixed_cue_picker import FixedCuePicker
from left_or_right import TrialSide
from task_stages import MAX_STAGE, STAGES
from tower_task import TowersTask
from village.scripts.log import log


class TowersTaskV3(TowersTask):
    """TowersTask with a fixed-N, fixed-position stimulus."""

    def start(self):
        self.settings.stage = MAX_STAGE  # start at the final stage for V3
        super().start()
        self.led_picker = FixedCuePicker(
            n_cues=int(getattr(self.settings, "n_cues", 10)),
            p=float(getattr(self.settings, "cue_p", 0.75)),
            start_dead_zone_cm=self.settings.led_start_dead_zone_cm,
            end_dead_zone_cm=self._odc.difficulty.end_dead_zone_cm)
        self._apply_stage(MAX_STAGE)
        if self._odc.phase == "warmup":
            self.led_picker.update_mu(self._odc.difficulty.mu_r, 0.0)
        log.info(f"[v3] n_cues={self.led_picker.n_cues} p={self.led_picker.p}"
                 f", cues every {self.led_picker.cue_separation_cm:.2f}cm, "
                 f"led_ms={self._odc.difficulty.led_ms}")
        self._update_hud()

    def _apply_stage(self, stage: int) -> None:
        """As TowersTask, minus the tower densities."""
        cfg = STAGES[stage]
        self.trial_is_cued = cfg.trial_is_cued
        self.give_free_reward = cfg.give_free_reward
        self.led_picker.update_dead_zone(self._odc.difficulty.end_dead_zone_cm)
        self.settings.stage = stage
        self.settings.checkpoint = self._odc.checkpoint
        self.settings.checkpoint_floor = self._odc.checkpoint_floor

    def _update_hud(self) -> None:
        super()._update_hud()
        n_r = len(self._this_trial_leds[TrialSide.RIGHT])
        n_l = len(self._this_trial_leds[TrialSide.LEFT])
        if n_r + n_l:
            self.cam_box.items_to_draw["hud"]["adv_label"].append(
                ("Cues R/L:", f" {n_r}/{n_l}", True))

    def after_trial(self):
        super().after_trial()
        self.register_value("n_cues", self.led_picker.n_cues)
        self.register_value("cue_p", self.led_picker.p)
