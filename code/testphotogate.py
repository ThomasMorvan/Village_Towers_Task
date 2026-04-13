from village.custom_classes.task import Event, Output, Task


class Test_Photogates(Task):

    def __init__(self):
        super().__init__()

        self.info = """Checking photogates."""

    def start(self):
        pass

    def close(self):
        pass

    def create_trial(self):
        self.bpod.add_state(
            state_name='Waiting',
            state_timer=300,
            state_change_conditions={Event.Port2In: 'Cross_corridor',
                                     Event.Tup: 'exit'},
            output_actions=[])

        self.bpod.add_state(
            state_name='Cross_corridor',
            state_timer=300,
            state_change_conditions={Event.Port2Out: 'Waiting',
                                     Event.Tup: 'exit'},
            output_actions=[(Output.PWM2, self.settings.light_intensity_high)])

    def after_trial(self):
        pass
