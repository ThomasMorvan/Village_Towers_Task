from village.custom_classes.task import Event, Output, Task
import random


# click on the link below to see the documentation about how to create
# tasks, plots and training protocols
# https://braincircuitsbehaviorlab.github.io/village/user_guide/create.html


class FollowTheLight(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Follow The Light Task
        -------------------

        This task is a simple visual task where the mouse has
        to poke the center port to start a trial.
        After the center port is poked,
        one of the two side ports will be illuminated.
        If the mouse licks the correct side port, it receives a reward.
        If the mouse licks the wrong side port, it receives a punishment.

        It contains 2 training stages:
        - Training stage 1: Only one side port is illuminated and gives reward.
                            No punishment is given, and the mouse can choose again.
        - Training stage 2: Both ports are illuminated with different intensity.
                            Brighter port gives reward, the other one gives punishment.

        The progression through the stages is defined in the training_settings.py file.
        """

    def start(self):
        """
        Al the variables created in training_protocol.py are accessible.
        - self.settings.reward_amount_ml: reward volume
        - self.settings.stage: current training stage


        - self.settings.light_intensity_high: high light intensity
        - self.settings.light_intensity_low: low light intensity
        - self.settings.trial_types: possible trial types
        - self.settings.punishment_time: punishment duration
        - self.settings.iti_time: inter-trial interval
        """

        # First we calculate the time that the valves (or pumps) need to open to deliver
        # the reward amount
        # Make sure to calibrate the valves before using this function, otherwise
        # it will return an Exception
        self.left_valve_opening_time = self.water_calibration.get_valve_time(
            port=1, volume=self.settings.reward_amount_ml
        )
        self.right_valve_opening_time = self.water_calibration.get_valve_time(
            port=3, volume=self.settings.reward_amount_ml
        )

        # determine if punishment will be used depending on stage
        if self.settings.stage == 1:
            # no punishment is used, let the mouse choose again
            self.punish_condition = "stimulus_state"
        else:
            # punishment is used
            self.punish_condition = "punish_state"

    def create_trial(self):
        # Pick a trial type at random
        self.this_trial_type = random.choice(self.settings.trial_types)

        # Set the variables for the stimulus states and the possible choices
        # based on the trial type
        self.stimulus_state_output = []
        if "left" in self.this_trial_type:
            self.stimulus_state_output.append(
                (Output.PWM1, self.settings.light_intensity_high)
            )
            if "hard" in self.this_trial_type:
                self.stimulus_state_output.append(
                    (Output.PWM3, self.settings.light_intensity_low)
                )
            self.left_poke_action = "reward_state"
            self.valve_to_open = Output.Valve1
            self.valve_opening_time = self.left_valve_opening_time
            self.right_poke_action = self.punish_condition

        elif "right" in self.this_trial_type:
            self.stimulus_state_output.append(
                (Output.PWM3, self.settings.light_intensity_high)
            )
            if "hard" in self.this_trial_type:
                self.stimulus_state_output.append(
                    (Output.PWM1, self.settings.light_intensity_low)
                )
            self.left_poke_action = self.punish_condition
            self.right_poke_action = "reward_state"
            self.valve_to_open = Output.Valve3
            self.valve_opening_time = self.right_valve_opening_time

        # 'ready_to_initiate' state that waits for the poke in the middle port
        self.bpod.add_state(
            state_name="ready_to_initiate",
            state_timer=0,
            state_change_conditions={Event.Port2In: "stimulus_state"},
            output_actions=[(Output.PWM2, self.settings.light_intensity_high)],
        )

        # 'stimulus_state' lights the side ports
        self.bpod.add_state(
            state_name="stimulus_state",
            state_timer=self.settings.timer_for_response,
            state_change_conditions={
                Event.Port1In: self.left_poke_action,
                Event.Port3In: self.right_poke_action,
                Event.Tup: "exit",
            },
            output_actions=self.stimulus_state_output,
        )

        # 'reward_state' delivers the reward
        self.bpod.add_state(
            state_name="reward_state",
            state_timer=self.valve_opening_time,
            state_change_conditions={Event.Tup: "iti_state"},
            output_actions=[self.valve_to_open],
        )

        # 'punish_state' waits during the punishment time
        self.bpod.add_state(
            state_name="punish_state",
            state_timer=self.settings.punishment_time,
            state_change_conditions={Event.Tup: "iti_state"},
            output_actions=[],
        )

        # 'iti_state' waits for certain time before starting the next trial
        # (inter-trial interval)
        self.bpod.add_state(
            state_name="iti_state",
            state_timer=self.settings.iti_time,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

    def after_trial(self):
        # First, we calculates the performance of a trial, comparing the trial type
        # to the first port that the mouse poked.
        # We can access the trial information in self.trial_data

        # get the side port that the mouse poked first
        first_poke = self.find_first_occurrence(
            self.trial_data["ordered_list_of_events"],
            ["Port1In", "Port3In"],
        )
        # check if the mouse poked the correct port
        if first_poke == "Port1In" and "left" in self.this_trial_type:
            correct = True
        elif first_poke == "Port3In" and "right" in self.this_trial_type:
            correct = True
        else:
            correct = False

        # register the amount of water given to the mouse in this trial
        # (this is always mandatory)
        self.register_value("water", self.settings.reward_amount_ml)

        # we will also record the trial type
        self.register_value("trial_type", self.this_trial_type)

        # we will also record if the trial was correct or not
        self.register_value("correct", correct)

    def close(self):
        pass

    def find_first_occurrence(self, event_list, targets):
        """
        Helper function to find the first occurrence of any target event in the list.

        Args:
            event_list: List of events
            targets: List of target events to look for

        Returns:
            The first target event found, or "NaN" if none are found
        """
        for event in event_list:
            if event in targets:
                return event
        return "NaN"
