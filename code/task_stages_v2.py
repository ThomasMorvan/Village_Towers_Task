"""Stage table for the V2 (speed-gated) Towers task."""

from task_stages import StageConfig, StagePolicy

GATES_CM_S = (55.0, 50.0, 45.0, 40.0, 35.0, 30.0)
SLOW_FRAC = 0.75


def _stage(i, gate):
    last = i == len(GATES_CM_S) - 1
    return StageConfig(
        stage=i, name=f"Speed{gate:.0f}" if not last else "Speed-Final",
        rwd_density=7.7, no_rwd_density=2.3,
        trial_is_cued=False, give_free_reward=True, both_sides_rewarded=False,
        staircases=(), color="darkorange" if not last else "orangered",
        advance_threshold=0.0 if last else 0.70, timed_leds=True,
        has_warmup=True, warmup_min_trials=10,
        warmup_acc_threshold=0.85, warmup_bias_threshold=0.10,
        policy=StagePolicy(jackpot=False if last else True))


STAGES_V2: dict[int, StageConfig] = {i: _stage(i, g)
                                     for i, g in enumerate(GATES_CM_S)}
STAGE_GATE: dict[int, float] = dict(enumerate(GATES_CM_S))
MIN_STAGE_V2 = 0
MAX_STAGE_V2 = len(GATES_CM_S) - 1
