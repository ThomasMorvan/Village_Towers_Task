from village.custom_classes.training_protocol_base import TrainingProtocolBase
from task_stages import MIN_STAGE, MAX_STAGE


class TrainingProtocol(TrainingProtocolBase):
    """Adaptive training protocol for the Towers Task.

    Rolling accuracy window: acc_window trials (default 40, same as paper).
    Stage transitions S1-S4 are checked within-session after every trial.
    Stage 0 is checked between sessions.

    Stage 0: BackForth --> learn to cross the maze back and forth for reward
        Both ports rewarded, no towers (mu_r=0, mu_nr=0), LEDs always on.
        Advance: last s0_required_sessions (default 2) sessions each have
        >=40 trials AND completion rate >=90% (trial complete = made a poke in
        ports 1 or 3, very easy in our case but consistent with paper).

    Stage 1: OneSide --> learn that towers cue the reward side
        One port rewarded per trial, rewarded towers only (mu_r=8.4, mu_nr=0),
        LEDs always on. No progressive difficulty.
        Advance: rolling acc >=80% AND side bias <=10% over acc_window trials.

    Stage 2: +mu_nr --> learn to deal with tower density distractors
        mu_r=8.4, LEDs always on. Kaernbach staircase increases mu_nr.
        Start: mu_nr=0. Target: mu_nr>=1.6 (paper eases mu_r 8.4->8.0 across
        T7->T9, but I hold mu_r=8.4 since the difference is negligible).
        delta_up=0.0025, delta_down=0.0075 (3:1 ratio -> p*=75%).
        Run-length escalation: step * r^(n-1), r=1.5, cap delta_max=0.025.
        Starting here, each session starts with a warmup phase with mu_nr=0
        (one-sided easy trials). Stage 2 uses a longer/looser gate (paper T8):
        warmup_min_trials=20 at >=80% correct, bias <=10%; stages 3-5 use 10
        trials at >=85%. All per-stage in STAGES. Then main phase begins.
        Onset multiplier M*exp(-t/tau)+1 (M=4.0, tau=10) is applied to the
        first onset_boost_trials (30) main-phase trials, then decays to 1.
        Disable that by setting M=0.
        Easy block (rescue): if rolling acc <65%, next 10 trials run mu_nr=0.
        Advance: rolling acc >=75% AND mu_nr>=1.6 over acc_window trials.

    Stage 3: -LED_ms --> learn to deal with timed LED cues
        mu_r=8.0, mu_nr fixed at 1.6 (S2 final value, paper T9). LEDs timed:
        each LED fires for led_ms then turns off. Staircase drives led_ms down.
        Warmup and easy-block trials stay untimed (always-on; paper T4/T7).
        Start: led_ms=5000.
        Target: led_ms<=200 (min_tower_duration).
        delta_up=10ms, p*=70% so delta_down=23.3ms (2.33:1), cap at 100ms.
        Run-length escalation as S2; warmup gate 10 trials at >=85%.
        Easy block (rescue): if rolling acc <60%, next 10 trials run mu_nr=0.
        Advance: rolling acc >=70% AND led_ms<=min_tower_duration.

    Stage 4: +mu_nr_short --> increase density at short LED (paper T10 -> T11)
        mu_r=7.7, timed LEDs fixed at min_tower_duration (200ms).
        Staircase drives mu_nr up (same step sizes as S2 but p*=70%, so
        delta_down=delta_up*0.70/0.30). Warmup/easy trials untimed; warmup
        gate 10 trials at >=85%.
        Start: mu_nr=1.6.
        Target: mu_nr>=2.3 (paper T10-> T11: 8:1.6 -> 7.7:2.3).
        Easy block (rescue): if rolling acc <60%, next 10 trials run mu_nr=0.
        Advance: rolling acc >=70% AND mu_nr>=2.3 over acc_window trials.

    Stage 5: Final --> paper T11
        mu_r=7.7, mu_nr=2.3 fixed, led_ms=200ms fixed. LEDs timed.
        No staircase. Easy block (rescue): if rolling acc drops below 55%, next
        rescue_block_size (10) trials run mu_nr=0 (untimed) before returning to
        full difficulty.
        No advancement criterion.

    Staircase equilibrium (per stage):
        Each staircase converges to its target_acc p* (S2 0.75, S3/S4 0.70),
        set per-stage in STAGES; delta_down = delta_up * p*/(1-p*).
    """

    def __init__(self) -> None:
        super().__init__()

    def default_training_settings(self) -> None:
        self.settings.next_task = "TowersTask"
        self.settings.refractory_period = 3600 * 4
        self.settings.minimum_duration = 600
        self.settings.maximum_duration = 3600

        # Stage and checkpoint tracking
        self.settings.stage = 0
        self.settings.checkpoint = 0
        self.settings.checkpoint_floor = 0.0
        self.settings.s0_valid_sessions = 0
        self.settings.s0_required_sessions = 2

        # Difficulty parameters (m^-1 scale for stage 2)
        #      correct +Δ     v_____________
        #             v_______|             |
        #      v______|                     | incorrect -3Δ
        # _____|                            |________________
        #
        # Pinto 2018 pace estimation:
        # T8: 4 sessions at ~ 150 trials/session --> 600 trials total
        # T9: 3.5 sessions at ~ 150 trials/session --> 525 trials total
        #
        # T8 tower density from 8.2:0 to 8.3:0.7   --> Δ = 0.7
        # T9 tower density from 8.2:0.7 to 8.3:1.6   --> Δ = 0.9
        #
        # essentially that's this more mu_nr per trial:
        # 0.7 / 600 = 0.0012
        # 0.9 / 525 = 0.0017
        #
        # so we pick something in between (0.0015/trial) for the
        # staircase step sizes. For a window of 40 trials, that corresponds
        # to 0.06 per window for incorrect at paper pace. If we take 0.1
        # instead, that's 1.7x faster, which seems reasonable for training.
        #
        # So for mu_nr, we have, at Δwindow = 0.1 / 40 trials:
        #     correct +0.0025 (0.1Δ / 40-trial window)
        #     incorrect -0.0075 (3x bigger)
        #     max 0.025 (10x bigger?)

        # At same ~speed going from 5000 ms to 200 ms over 600 trials is about
        # (5000-200) / 600 = 8.0 ms per trial, or 320 ms per 40-trial window.
        # So we pick something a bit faster for training, let's say 10ms
        # per trial, which is 400 ms per window. That corresponds to:
        #     correct -10 ms (400Δ ms / 40-trial window)
        #     incorrect +30 ms (3x bigger)
        #     max 100 ms (10x bigger?)
        # Note: I changed the incorrect step size. Instead of fixed 3x bigger,
        # which is correct when the target accuracy is 75%, it is now computed
        # based on the target accuracy. So we have:
        #     delta_down = delta_up * p/(1-p)
        #   previously: delta_up * 0.75/0.25 = delta_up * 3
        #
        self.settings.staircase_delta_up = 0.0025  # m^-1 per correct
        # p* (target accuracy) is per-stage: Staircase.target_acc in STAGES.
        # delta_down = delta_up * p*/(1-p*).
        self.settings.staircase_r = 1.5  # step scaling factor
        self.settings.staircase_M = 4.0  # main-phase onset multiplier
        self.settings.staircase_tau = 10.0  # main-phase onset tau (trials)
        self.settings.onset_boost_trials = 30  # main-phase trials boost
        self.settings.onset_boost_on_graduation = False  # boost at chkpt
        self.settings.staircase_delta_max = 0.025  # max step size

        # Stage 3 staircase parameters (ms scale), same idea, but we go down
        self.settings.staircase_delta_up_ms = 10  # ms per correct
        self.settings.staircase_delta_max_ms = 100  # ms, max step size
        self.settings.min_tower_duration = 200  # ms, led ms target

        # Task geometry
        self.settings.led_start_dead_zone_cm = 10
        self.settings.acc_window = 40  # rolling accuracy window (trials)
        self.settings.rescue_enabled = True
        self.settings.rescue_block_size = 10
        self.settings.resume_from_last = True  # resume last session difficulty

        # Reward policy (jackpot and effort scaling; see reward_policy.py)
        self.settings.reward_policy_enabled = False  # master switch
        self.settings.reward_effort_max_mult = 2.0
        self.settings.reward_effort_delta_easy = 6.0  # delta with no bonus
        self.settings.reward_jackpot_mult = 10.0
        self.settings.reward_jackpot_prob = 0.1  # of correct trials

        # Input/output settings
        self.settings.reward_amount_ml = 0.08
        self.settings.light_intensity_high = 255
        self.settings.light_intensity_low = 50

    def update_training_settings(self) -> None:
        df_task = self.df[self.df["task"] == "TowersTask"]
        df_task = df_task.dropna(subset=["stage"])
        if df_task.empty:
            return

        if self.settings.stage == 0:
            # Advance after 2 sessions with >=40 trials AND >=90% completion.
            # trial_correct is NaN on timeouts; notna().mean() = completion %
            def _ok(s):
                df_s = df_task[df_task["session"] == s]
                return (len(df_s) >= 40
                        and df_s["trial_correct"].notna().mean() >= 0.90)

            all_sessions = sorted(df_task["session"].unique())
            n_valid = int(sum(_ok(s) for s in all_sessions))
            self.settings.s0_valid_sessions = n_valid

            n_req = int(self.settings.s0_required_sessions)
            recent = all_sessions[-n_req:]
            if len(recent) == n_req and all(_ok(s) for s in recent):
                self.settings.stage = 1
                self.settings.checkpoint = 1
                self.settings.checkpoint_floor = 0.0
                print("   * [TrainingProtocol] Stage 0 -> 1")
        else:
            if self.last_task != "TowersTask":
                return
            # Stages 1-3: checkpoints handled within-session.
            # Restore last known stage/floor; never (?) regress below current.
            last_row = df_task.sort_values(["session", "trial"]).iloc[-1]
            restored = int(min(MAX_STAGE, max(MIN_STAGE, last_row["stage"])))
            self.settings.stage = int(max(self.settings.stage, restored))
            self.settings.checkpoint = int(last_row.get("checkpoint", 0))
            self.settings.checkpoint_floor = float(last_row.get(
                "checkpoint_floor", last_row.get("mu_nr", 0.0)))
            self.settings.last_mu_nr = float(last_row.get("mu_nr", 0.0))
            self.settings.last_led_ms = int(last_row.get("led_ms", 5000))

    def define_gui_tabs(self) -> None:
        self.gui_tabs = {
            "Reward and reward policy": [
                "reward_amount_ml",
                "light_intensity_high",
                "light_intensity_low",
                "reward_policy_enabled",
                "reward_jackpot_mult",
                "reward_jackpot_prob",
                "reward_effort_max_mult",
                "reward_effort_delta_easy",
            ],
            "Task": [
                "next_task",
                "refractory_period",
                "minimum_duration",
                "maximum_duration",
                "led_start_dead_zone_cm",
                "acc_window",
                "rescue_enabled",
                "rescue_block_size",
            ],
            "Stage": [
                "stage",
                "checkpoint",
                "checkpoint_floor",
                "s0_required_sessions",
                "resume_from_last",
            ],
            "Staircase": [
                "staircase_delta_up",
                "staircase_delta_up_ms",
                "staircase_r",
                "staircase_delta_max",
                "staircase_delta_max_ms",
                "staircase_M",
                "staircase_tau",
                "onset_boost_trials",
                "onset_boost_on_graduation",
                "min_tower_duration",
            ],
        }
