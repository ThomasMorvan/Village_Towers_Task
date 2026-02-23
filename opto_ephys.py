from village.custom_classes.task import Event, Output, Task



class OptoEphys(Task):

    def __init__(self):
        super().__init__()


    def start(self):
        pass


    def create_trial(self):

        self.bpod.add_state(
            state_name="Off",
            state_timer=10,
            state_change_conditions={Event.Tup: "On"},
            output_actions=[Output.BNC1Low, Output.BNC2Low],
        )

        self.bpod.add_state(
            state_name="On",
            state_timer=4,
            state_change_conditions={Event.Tup: "Off"},
            output_actions=[Output.BNC1High, Output.BNC2High],
        )


    def after_trial(self):
        pass


    def close(self):
        pass
