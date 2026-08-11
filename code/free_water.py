from village.custom_classes.task_base import (BpodEvent as Event,
                                              BpodOutput as Output, TaskBase)


class Free_Water(TaskBase):
    """Poke any port, get water. No trial structure, no timeout.

    Meant as a failsafe session when a mouse isn't drinking enough: all of the
    3 ports light up and deliver reward on poke, forever, until the
    session's max duration. If one port is physically broken the mouse
    can just use the other two.
    """
    MINIMUM_DURATION_S = 180
    MAXIMUM_DURATION_S = 300
    REFRACTORY_PERIOD_S = 4 * 3600

    def __init__(self):
        super().__init__()
        self.info = """

        Free Water
        -------------------

        Poke any illuminated port to get a water reward. No trial
        structure, no penalties. Use to make sure a mouse is drinking.
        """

    def start(self):
        self.settings.minimum_duration = self.MINIMUM_DURATION_S
        self.settings.maximum_duration = self.MAXIMUM_DURATION_S
        self.settings.refractory_period = self.REFRACTORY_PERIOD_S
        water_cal = self.calibrations.bpod_water_calibration
        self.valve_opening_time = {port: water_cal.get_valve_time(
            port=port, volume=self.settings.big_reward_amount_ul)
            for port in (1, 2, 3)}

    def create_trial(self):
        self.bpod.add_state(
            state_name='Waiting',
            state_timer=300,
            state_change_conditions={Event.Port1In: 'Reward_1',
                                     Event.Port2In: 'Reward_2',
                                     Event.Port3In: 'Reward_3',
                                     Event.Tup: 'exit'},
            output_actions=[(Output.PWM1, self.settings.light_intensity_high),
                            (Output.PWM2, self.settings.light_intensity_high),
                            (Output.PWM3, self.settings.light_intensity_high)])

        self.bpod.add_state(
            state_name='Reward_1',
            state_timer=self.valve_opening_time[1],
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[Output.Valve1])

        self.bpod.add_state(
            state_name='Reward_2',
            state_timer=self.valve_opening_time[2],
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[Output.Valve2])

        self.bpod.add_state(
            state_name='Reward_3',
            state_timer=self.valve_opening_time[3],
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[Output.Valve3])

    def after_trial(self):
        poked_port = next(
            (p for p, e in ((1, 'Port1In'), (2, 'Port2In'), (3, 'Port3In'))
             if e in self.trial_data["ordered_list_of_events"]), None)
        self.register_value("water",
                            self.settings.big_reward_amount_ul if poked_port
                            else 0.0)

    def close(self):
        pass
