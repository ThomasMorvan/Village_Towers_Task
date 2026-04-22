import random
import pandas as pd

from village.custom_classes.auto_no_mouse_base import AutoNoMouse_Base
from left_or_right import TrialSide


class AutoNoMouse(AutoNoMouse_Base):
    """Drives TowersTask automatically as a virtual animal.

    Args:
        task: The running TowersTask instance.
        accuracy_left:  Probability [0-1] of correct choice when reward is LEFT.
        accuracy_right: Probability [0-1] of correct choice when reward is RIGHT.
    """

    # Corridor traversal
    X_ENTRY: int = 630        # rightmost x (entry side)
    X_FAR: int = 20           # leftmost x (far/choice end)
    X_STEP: int = 5           # pixels per step during traversal
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

    def __init__(self, task,
                 accuracy_left: float = 0.75,
                 accuracy_right: float = 0.75,
                 ) -> None:
        super().__init__(task)
        self.accuracy_left = accuracy_left
        self.accuracy_right = accuracy_right

    def _accuracy_for(self, side: TrialSide) -> float:
        return (self.accuracy_left if side == TrialSide.LEFT
                else self.accuracy_right)

    def clip_y(self, y: int) -> int:
        return max(self.Y_CLIP_MIN, min(self.Y_CLIP_MAX, y))

    def clip_x(self, x: int) -> int:
        return max(self.X_CLIP_MIN, min(self.X_CLIP_MAX, x))

    def run_trial(self) -> None:
        """One complete simulated trial: poke → traverse → choose → return."""
        self.wait(self.WAIT_START)
        if self._stop_event.is_set():
            return

        # 1. Traverse corridor right to left
        y_prev = self.Y_CENTER
        for x in range(self.X_ENTRY, self.X_FAR, -self.X_STEP):
            if self._stop_event.is_set():
                return
            y = self.clip_y(y_prev + random.randint(-self.Y_JITTER,
                                                    self.Y_JITTER))
            self.set_position(x, y)
            y_prev = y
            self.task.softcode_callback()
            self.wait(self.FRAMES_PER_STEP / self.FPS)

        # 2. Choose side, move, and poke
        correct_side = self.task.current_trial_rwd_side
        if random.random() < self._accuracy_for(correct_side):
            chosen = correct_side
        else:
            chosen = (TrialSide.RIGHT if correct_side == TrialSide.LEFT
                      else TrialSide.LEFT)
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

    def inject_trial(self, p_correct_left: float = 1.0,
                     p_correct_right: float = 1.0) -> None:
        """Append one mock trial row directly to task.session_df."""
        task = self.task

        side = task.left_or_right.draw_next_trial()
        rwd_leds, no_rwd_leds = task.led_picker.draw_towers()
        if len(rwd_leds) < len(no_rwd_leds):
            rwd_leds, no_rwd_leds = no_rwd_leds, rwd_leds

        p_correct = p_correct_left if side == TrialSide.LEFT else p_correct_right
        correct = random.random() < p_correct
        chosen = side if correct else (
            TrialSide.RIGHT if side == TrialSide.LEFT else TrialSide.LEFT
        )
        choice_port = "Port1In" if chosen == TrialSide.LEFT else "Port3In"

        t0 = 0.0
        t_start = round(random.uniform(*self.T_START_RANGE), 3)
        t_leds = round(t_start + random.uniform(*self.T_LEDS_RANGE), 3)
        t_choice = round(t_leds + random.uniform(*self.T_CHOICE_RANGE), 3)
        t_end = round(t_choice + random.uniform(*self.T_END_RANGE), 3)

        left_leds = (rwd_leds if side == TrialSide.LEFT else no_rwd_leds).tolist()
        right_leds = (rwd_leds if side == TrialSide.RIGHT else no_rwd_leds).tolist()

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
            "water": task.settings.reward_amount_ml if correct else 0,
            "trial_correct": correct,
        }

        task.session_df = pd.concat([task.session_df, pd.DataFrame([row])],
                                    ignore_index=True)
        task.current_trial += 1
