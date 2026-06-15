import random as _random


class RewardPolicy:
    """Optional reward size boost for training stages trials.
    Need to activate the master switch in settings (reward_policy_enabled) and
    then specify which modes to apply per stage in the
    StagePolicy (STAGES[stage].policy). Modes:
    1) Jackpots (from Gong, Martell, Dudman & Coddington 2026):
        a fixed large multiplier (10x) on 10% of trials
    2) Effort scaling: a bonus for correct responses on harder trials
        (e.g. smaller delta_towers). Ramps from 1x at "easy" threshold up to
        effort_max_mult for the hardest trials (in settings).
    """
    def __init__(self, *, enabled: bool = False, jackpot_mult: float = 10.0,
                 jackpot_prob: float = 0.2, effort_max_mult: float = 2.0,
                 effort_delta_easy: float = 6.0, rng=None) -> None:
        self.enabled = bool(enabled)
        self.jackpot_mult = float(jackpot_mult)
        self.jackpot_prob = float(jackpot_prob)
        self.effort_max_mult = float(effort_max_mult)
        self.effort_delta_easy = float(effort_delta_easy)
        self._rng = rng if rng is not None else _random.Random()
        self.last_was_jackpot = False

    @classmethod
    def from_settings(cls, settings, rng=None) -> "RewardPolicy":
        def g(key, default):
            return getattr(settings, key, default)
        return cls(enabled=g("reward_policy_enabled", False),
                   jackpot_mult=g("reward_jackpot_mult", 10.0),
                   jackpot_prob=g("reward_jackpot_prob", 0.2),
                   effort_max_mult=g("reward_effort_max_mult", 2.0),
                   effort_delta_easy=g("reward_effort_delta_easy", 6.0),
                   rng=rng)

    def reward_mult_for_trial(self, *, jackpot: bool = False,
                              effort: bool = False,
                              delta_towers: float | None = None,
                              single_sided: bool = True,
                              main_phase: bool = True) -> float:
        self.last_was_jackpot = False
        if not self.enabled or not single_sided or not main_phase:
            # If both sides are rewarded, does not apply (stage 0).
            # If not main phase, does not apply (warmup, rescue).
            return 1.0

        # Jackpot trials
        if jackpot and self._rng.random() < self.jackpot_prob:
            self.last_was_jackpot = True
            return self.jackpot_mult

        # Effort scaling: bonus for hard trials
        if (effort
                and delta_towers is not None
                and delta_towers > 0
                and self.effort_delta_easy > 0):
            frac = max(0.0, 1.0 - delta_towers / self.effort_delta_easy)
            return 1.0 + (self.effort_max_mult - 1.0) * frac

        return 1.0
