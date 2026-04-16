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
            state_change_conditions={Event.Port1In: 'Poke_1',
                                     Event.Port2In: 'Poke_2',
                                     Event.Port3In: 'Poke_3',
                                     Event.Tup: 'exit'},
            output_actions=[])

        self.bpod.add_state(
            state_name='Poke_1',
            state_timer=300,
            state_change_conditions={Event.Port1Out: 'Waiting',
                                     Event.Tup: 'exit'},
            output_actions=[(Output.PWM1, self.settings.light_intensity_high),
                            Output.SoftCode6])

        self.bpod.add_state(
            state_name='Poke_2',
            state_timer=300,
            state_change_conditions={Event.Port2Out: 'Waiting',
                                     Event.Tup: 'exit'},
            output_actions=[(Output.PWM2, self.settings.light_intensity_high),
                            Output.SoftCode7])

        self.bpod.add_state(
            state_name='Poke_3',
            state_timer=300,
            state_change_conditions={Event.Port3Out: 'Waiting',
                                     Event.Tup: 'exit'},
            output_actions=[(Output.PWM3, self.settings.light_intensity_high),
                            Output.SoftCode8])

    def _print(self, idx):
        print(f"Input in {idx}")

    def after_trial(self):
        pass
