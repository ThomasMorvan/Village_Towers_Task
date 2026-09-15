"""Stage table for the V2 (speed-gated) Towers task."""

from task_stages import StageConfig, StagePolicy, Staircase

START_SPEED_CM_S = 60.0
TARGET_SPEED_CM_S = 30.0

STAGES_V2: dict[int, StageConfig] = {
    0: StageConfig(
        stage=0, name="SpeedShaping",
        rwd_density=7.7, no_rwd_density=2.3,
        trial_is_cued=False, give_free_reward=True,
        both_sides_rewarded=False,
        staircases=(Staircase(variable="max_speed",
                              start=START_SPEED_CM_S,
                              target=TARGET_SPEED_CM_S,
                              harder_direction="down",
                              target_acc=0.70),),
        color="darkorange",
        advance_threshold=0.70,
        timed_leds=True,
        has_warmup=True, warmup_min_trials=10,
        warmup_acc_threshold=0.85, warmup_bias_threshold=0.10,
        policy=StagePolicy(jackpot=True)),
    1: StageConfig(
        stage=1, name="SpeedFinal",
        rwd_density=7.7, no_rwd_density=2.3,
        trial_is_cued=False, give_free_reward=True,
        both_sides_rewarded=False,
        staircases=(),
        color="orangered",
        advance_threshold=0.0,
        timed_leds=True,
        has_warmup=True, warmup_min_trials=10,
        warmup_acc_threshold=0.85, warmup_bias_threshold=0.10,
        policy=StagePolicy()),
}

MIN_STAGE_V2 = 0
MAX_STAGE_V2 = 1
