import random

import numpy as np
import pandas as pd

from village.custom_classes.auto_no_mouse_base import AutoNoMouseBase, \
                                                      AutonomouseParam
from village.custom_classes.task import Task
from village.scripts.time_utils import time_utils
from left_or_right import TrialSide, TrialResult
from task_stages import STAGES
from decision_maker import DecisionMaker


class AutoNoMouse(AutoNoMouseBase):
    """Drives TowersTask automatically as a virtual animal.

    Args:
        task: The running TowersTask instance.
    """

    TASK_NAME = "TowersTask"

    PARAMS = [AutonomouseParam("p_correct_left", float, 0.75, "p correct L",
                               0.0, 1.0, "P(correct | reward LEFT)"),
              AutonomouseParam("p_correct_right", float, 0.75, "p correct R",
                               0.0, 1.0, "P(correct | reward RIGHT)"),
              AutonomouseParam("use_ddm", bool, True, "Use DDM",
                               0, 1, "1=DDM accumulator, 0=flat accuracy"),
              AutonomouseParam("ddm_plot", bool, False, "Plot DDM",
                               0, 1, "1=Plot DDM acc (debug), 0=Do not plot")]

    # Corridor traversal
    X_ENTRY: int = 630        # rightmost x (entry side)
    X_FAR: int = 20           # leftmost x (far/choice end)
    X_STEP: int = 5           # pixels per step during traversal
    CORRIDOR_LEN_M: float = 1.2  # physical corridor length (metres)

    # median DDM parameters from Pinto 2018
    # sigma2_s converted from towers²/m to towers²/tower by dividing
    # by reward density 7.7 towers/m)
    ACC_THETA: np.ndarray = np.array([
        0.128,        # sigma2_i
        8.0,          # B
        -0.328,       # lam
        0.143,        # sigma2_a
        66 / 7.7,     # sigma2_s
        0.154,        # phi
        0.065,        # tau_phi
        -0.055,       # bias
        0.06,         # lapse
    ])

    # Autonomouse with less noise (so better performance) for debug
    ACC_THETA: np.ndarray = np.array([.1, 8, -0.2, .1, 20 / 7.7,
                                      0.154, 0.065, 0, 0])

    X_RETURN_STEP: int = 5    # pixels per step on the return run
    X_JITTER: int = 5         # max random x deviation per step
    Y_CENTER: int = 220       # y when running straight through
    Y_CLIP_MIN: int = 230     # y corridor bounds
    Y_CLIP_MAX: int = 250
    X_CLIP_MIN: int = 10      # x corridor bounds
    X_CLIP_MAX: int = 30
    Y_JITTER: int = 5         # max random y deviation per step
    Y_PORT_LEFT: int = 400    # y position of left reward port
    Y_PORT_RIGHT: int = 100   # y position of right reward port
    Y_STEP: int = 5           # pixels per step during y movement
    FRAMES_PER_STEP: int = 1  # frames to wait between each step in movement

    # Timing
    FPS: float = 25.0        # simulated frame rate
    WAIT_PRE_POKE: float = 0.3
    WAIT_START: float = 0.3
    WAIT_POST_CHOICE: float = 0.5
    WAIT_END_TRIAL: float = 0.5

    # Inject timing ranges (uniform distribution bounds)
    T_START_RANGE: tuple = (0.2, 0.6)
    T_LEDS_RANGE: tuple = (0.3, 0.7)
    T_CHOICE_RANGE: tuple = (2.0, 8.0)
    T_END_RANGE: tuple = (0.2, 0.5)
    T_PORT2_OUT: float = 0.1     # Port2Out offset after Port2In
    T_REWARD_DUR: float = 0.2    # reward state duration
    T_NO_REWARD_DUR: float = 0.01

    def __init__(self, task: Task = None) -> None:
        super().__init__(task)
        self.acc = DecisionMaker(self.ACC_THETA, [], [])
        self._acc_steps = max(1, round(
            self.X_STEP * self.CORRIDOR_LEN_M
            / (self.X_ENTRY - self.X_FAR) / DecisionMaker.DT))
        self._last_draw: float | None = None

    def _choose_ddm(self) -> TrialSide:
        """Sample choice from the current accumulator state."""
        self._last_draw = random.random()
        return (TrialSide.RIGHT if self._last_draw < self.acc.p_right()
                else TrialSide.LEFT)

    def _choose_accuracy(self, rewarded_side: TrialSide) -> TrialSide:
        """Flat-accuracy choice using p_correct_left / p_correct_right."""
        self._last_draw = random.random()
        p = (self.p_correct_left if rewarded_side == TrialSide.LEFT
             else self.p_correct_right)
        return (rewarded_side if self._last_draw < p
                else (TrialSide.RIGHT if rewarded_side == TrialSide.LEFT
                      else TrialSide.LEFT))

    def clip_y(self, y: int) -> int:
        return max(self.Y_CLIP_MIN, min(self.Y_CLIP_MAX, y))

    def clip_x(self, x: int) -> int:
        return max(self.X_CLIP_MIN, min(self.X_CLIP_MAX, x))

    def run_trial(self) -> None:
        """One complete simulated trial:
        poke --> traverse --> choose --> return."""
        self.wait(self.WAIT_START)
        if self._stop_event.is_set():
            return

        # 1. Traverse corridor right to left and step the DDM accumulator.
        L_pos, R_pos = DecisionMaker.positions_from_task(
            self.task, self.X_ENTRY, self.X_FAR, self.CORRIDOR_LEN_M)
        self.acc.reset(L_pos, R_pos)
        y_prev = self.Y_CENTER
        for x in range(self.X_ENTRY, self.X_FAR, -self.X_STEP):
            if self._stop_event.is_set():
                return
            y = self.clip_y(y_prev + random.randint(-self.Y_JITTER,
                                                    self.Y_JITTER))
            self.set_position(x, y)
            y_prev = y
            self.task.softcode_callback()
            self.acc.step(n=self._acc_steps)
            self.wait(self.FRAMES_PER_STEP / self.FPS)

        if self.ddm_plot:
            self.acc.plot(L_pos, R_pos)

        # 2. Choose side via accumulator P(right), move, and poke.
        chosen = self._choose_ddm() if self.use_ddm else self._choose_accuracy(
            self.task.current_trial_rwd_side)
        port = 1 if chosen == TrialSide.LEFT else 3
        y_target = (self.Y_PORT_LEFT if chosen == TrialSide.LEFT
                    else self.Y_PORT_RIGHT)

        x_prev = self.X_FAR
        step = -self.Y_STEP if y_prev > y_target else self.Y_STEP
        for move_y in range(y_prev, y_target, step):
            if self._stop_event.is_set():
                return
            x = self.clip_x(x_prev + random.randint(-self.X_JITTER,
                                                    self.X_JITTER))
            x_prev = x
            self.set_position(x, move_y)
            self.wait(self.FRAMES_PER_STEP / self.FPS)
        self.set_position(x_prev, y_target)
        self.poke(port)
        self.wait(self.WAIT_POST_CHOICE)

        # 3. Return traversal: back to centre y, then right
        y_prev = y_target
        step = -self.Y_STEP if y_prev > self.Y_CENTER else self.Y_STEP
        for move_y in range(y_prev, self.Y_CENTER, step):
            if self._stop_event.is_set():
                return
            x = self.clip_x(x_prev + random.randint(-self.X_JITTER,
                                                    self.X_JITTER))
            x_prev = x
            self.set_position(x, move_y)
            y_prev = move_y
            self.wait(self.FRAMES_PER_STEP / self.FPS)

        for x in range(self.X_FAR, self.X_ENTRY + 1, self.X_RETURN_STEP):
            if self._stop_event.is_set():
                return
            y = self.clip_y(y_prev + random.randint(-self.Y_JITTER,
                                                    self.Y_JITTER))
            self.set_position(x, y)
            y_prev = y
            self.wait(self.FRAMES_PER_STEP / self.FPS)

        # 4. Initiate trial with poke in middle port
        self.wait(self.WAIT_PRE_POKE)
        self.poke(2)
        self.wait(self.WAIT_END_TRIAL)

    def inject_trial(self, **kwargs) -> None:
        """Append one mock trial row directly to task.session_df."""
        for param in self.PARAMS:
            if param.name in kwargs:
                setattr(self, param.name, param.clamp(kwargs[param.name]))
        task = self.task

        both_sides = STAGES[task._odc.stage].both_sides_rewarded
        if both_sides:
            side = TrialSide.BOTH
            pR = task.left_or_right.current_PR
            empR = task.left_or_right.current_empR
            draw_side = task.left_or_right.current_draw_side.value
            draw_prob = task.left_or_right.current_draw_prob
            rwd_leds = no_rwd_leds = np.array([], dtype=int)
            left_leds_idx = right_leds_idx = np.array([], dtype=int)
            chosen = random.choice([TrialSide.LEFT, TrialSide.RIGHT])
        else:
            side = task.left_or_right.draw_next_trial()
            pR = task.left_or_right.current_PR
            empR = task.left_or_right.current_empR
            draw_side = task.left_or_right.current_draw_side.value
            draw_prob = task.left_or_right.current_draw_prob

            rwd_leds, no_rwd_leds = task.led_picker.draw_towers()
            if len(rwd_leds) < len(no_rwd_leds):
                rwd_leds, no_rwd_leds = no_rwd_leds, rwd_leds

            left_leds_idx = rwd_leds if side == TrialSide.LEFT else no_rwd_leds
            right_leds_idx = (rwd_leds if side == TrialSide.RIGHT
                              else no_rwd_leds)
            L_pos, R_pos = DecisionMaker.positions_from_task(
                task, self.X_ENTRY, self.X_FAR, self.CORRIDOR_LEN_M,
                led_dict={TrialSide.LEFT: left_leds_idx,
                          TrialSide.RIGHT: right_leds_idx})

            if self.use_ddm and len(L_pos) + len(R_pos) > 0:
                steps = max(1, round(self.CORRIDOR_LEN_M / DecisionMaker.DT))
                self.acc.reset(L_pos, R_pos)
                self.acc.step(n=steps)
                chosen = self._choose_ddm()
            else:
                chosen = self._choose_accuracy(side)

        correct = True if both_sides else (chosen == side)
        choice_port = "Port1In" if chosen == TrialSide.LEFT else "Port3In"

        t0 = time_utils.now_timestamp()
        task.recorder.start_trial(t0, t0)
        t_start = round(t0 + random.uniform(*self.T_START_RANGE), 3)
        t_leds = round(t_start + random.uniform(*self.T_LEDS_RANGE), 3)
        t_choice = round(t_leds + random.uniform(*self.T_CHOICE_RANGE), 3)
        t_end = round(t_choice + random.uniform(*self.T_END_RANGE), 3)
        task.recorder.end_trial(t_end)

        left_leds = left_leds_idx.tolist()
        right_leds = right_leds_idx.tolist()

        row: dict = {
            "date": task.date,
            "trial": task.current_trial,
            "subject": task.subject,
            "task": task.name,
            "system_name": task.system_name,
            "TRIAL_START": t0,
            "TRIAL_END": t_end,
            "Port2In": [t_start],
            "Port2Out": [t_start + self.T_PORT2_OUT],
            choice_port: [t_choice],
            "ordered_list_of_events": ["Port2In", "Port2Out", choice_port],
            "STATE_START_START": [t0],
            "STATE_START_END": [t_start],
            "STATE_turn_on_leds_START": [t_leds],
            "STATE_turn_on_leds_END": [t_choice],
            "STATE_reward_state_START": [t_choice] if correct else [],
            "STATE_reward_state_END": (
                [round(t_choice + self.T_REWARD_DUR, 3)] if correct else []
            ),
            "STATE_no_reward_state_START": [] if correct else [t_choice],
            "STATE_no_reward_state_END": (
                [] if correct else [round(t_choice + self.T_NO_REWARD_DUR, 3)]
            ),
            "STATE_END_TRIAL_START": [round(t_end - self.T_NO_REWARD_DUR, 3)],
            "STATE_END_TRIAL_END": [t_end],
            "L LEDs": left_leds,
            "R LEDs": right_leds,
            "trial_side": side.value,
            "water": task.settings.reward_amount_ml,
            "trial_correct": correct,
            "rwd_density": task.led_picker.mu_reward,
            "no_rwd_density": task.led_picker.mu_no_reward,
            "pR": pR,
            "empR": empR,
            "draw_side": draw_side,
            "draw_prob": draw_prob,
            "stage": task._odc.stage,
            "phase": task._odc.phase,
            "mu_r": task._odc.difficulty.mu_r,
            "mu_nr": (0.0 if task._odc.phase == "warmup"
                      else task._odc.difficulty.mu_nr),
            "led_ms": task._odc.difficulty.led_ms,
            "checkpoint_floor": task._odc.checkpoint_floor,
            "streak": task._odc.streak,
            "checkpoint": task._odc.checkpoint,
            "warmup_trial": task._odc.warmup_n,
            "trial_is_cued": int(task.trial_is_cued),
            "give_free_reward": int(task.give_free_reward),
            "rescue": int(task._odc.rescue_active),
            "delta_towers": len(rwd_leds) - len(no_rwd_leds),
            "step_delta": 0.0,
            "step_boost": 1.0,
        }

        task.session_df = pd.concat([task.session_df, pd.DataFrame([row])],
                                    ignore_index=True)
        task.left_or_right.add_trial(TrialResult(side=side, correct=correct))
        task.current_trial_rwd_side = side
        task.is_trial_correct = correct
        task._after_trial_adaptation()
        task.current_trial_rwd_side = TrialSide.NONE
        task.session_df.at[task.session_df.index[-1], "step_delta"] = (
            task._odc.last_delta)
        task.session_df.at[task.session_df.index[-1], "step_boost"] = (
            task._odc.last_boost)
        task._update_hud()
        task.is_trial_correct = None
        task.current_trial += 1
