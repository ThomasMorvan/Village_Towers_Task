from village.custom_classes.task_base import (BpodEvent as Event,
                                         BpodOutput as Output, TaskBase)

class TaskTest(TaskBase):
    def __init__(self):
        super().__init__()

    def start(self):
        pass

    def create_trial(self):

        # load play stop
        # load play
        # load play stop stop
        # load play play stop
        # load load play stop
        # load(1) play stop
        # load(1) play
        # play stop
        # stop load play

        self.bpod.add_state(
            state_name="load",
            state_timer=1,
            state_change_conditions={Event.Tup: "play"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="play",
            state_timer=1,
            state_change_conditions={Event.Tup: "stop"},
            output_actions=[Output.SoftCode1],
        )

        self.bpod.add_state(
            state_name="stop",
            state_timer=1,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[Output.SoftCode1],
        )

    def after_trial(self):
        pass

    def close(self):
        pass
