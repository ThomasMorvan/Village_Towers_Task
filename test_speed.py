from village.custom_classes.task import Event, Output, Task


# click on the link below to see the documentation about how to create
# tasks, plots and training protocols
# https://braincircuitsbehaviorlab.github.io/village/user_guide/create.html


class TestSpeed(Task):
    """
    This class defines the task.

    Required methods to implement:
    - __init__: Initialize the task
    - start: Called when the task starts.
    - create_trial: Called once per trial to create the state machine.
    - after_trial: Called once after each trial to register the values in the .csv file.
    - close: Called when the task is finished.
    """

    def __init__(self):
        """
        Initialize the training protocol. The text in the self.info variable
        will be shown when the task is selected in the GUI to be run manually.
        """
        super().__init__()

        self.info = """

        Habituation Task
        -------------------

        This task is a simple visual task where the mouse has
        to poke in illuminated ports.
        The center port illuminates when a trial starts.
        After the center port is poked,
        both side ports are illuminated and give reward.
        """

    def start(self):
        """
        This function is called when the task starts.
        It is used to calculate values needed for the task.
        The following variables are accesible by default:
        - self.bpod: (Bpod object)
        - self.name: (str) the name of the task
                (it is the name of the class, in this case Habituation)
        - self.subject: (str) the name of the subject performing the task
        - self.current_trial: (int) the current trial number starting from 1
        - self.system_name: (str) the name of the system as defined in the
                                tab settings of the GUI
        - self.settings: (Settings object) the settings defined in training_protocol.py
        - self.trial_data: (dict) information about the current trial
        - self.force_stop: (bool) if made true the task will stop

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
        # self.left_valve_opening_time = self.water_calibration.get_valve_time(
        #     port=1, volume=self.settings.reward_amount_ml
        # )
        # self.right_valve_opening_time = self.water_calibration.get_valve_time(
        #     port=3, volume=self.settings.reward_amount_ml
        # )

    def create_trial(self):
        """
        This function is called once per trial, first it modifies variables and then
        sends the state machine to the bpod that will run the trial.
        """

        self.controller.add_state(
            state_name="A",
            state_timer=2,
            state_change_conditions={Event.Tup: "B"},
            output_actions=[Output.SoftCode7, Output.BNC1Low],
        )

        self.controller.add_state(
            state_name="B",
            state_timer=2,
            state_change_conditions={Event.Tup: "A"},
            output_actions=[Output.SoftCode14, Output.BNC1High],
        )

        # self.bpod.add_state(
        #     state_name="first",
        #     state_timer=1,
        #     state_change_conditions={Event.Tup: "second"},
        #     #output_actions=[Output.SoftCode1, Output.BNC1Low],
        #     output_actions=[Output.BNC1High, Output.BNC2Low],
        # )

        # self.bpod.add_state(
        #     state_name="second",
        #     state_timer=1,
        #     state_change_conditions={Event.Tup: "third"},
        #     #output_actions=[Output.SoftCode2, Output.BNC1High],
        #     output_actions=[Output.BNC1High, Output.BNC2High],
        # )

        # self.bpod.add_state(
        #     state_name="third",
        #     state_timer=1,
        #     state_change_conditions={Event.Tup: "fourth"},
        #     #output_actions=[Output.SoftCode2, Output.BNC1High],
        #     output_actions=[Output.BNC1Low, Output.BNC2High],
        # )

        # self.bpod.add_state(
        #     state_name="fourth",
        #     state_timer=1,
        #     state_change_conditions={Event.Tup: "fifth"},
        #     #output_actions=[Output.SoftCode2, Output.BNC1High],
        #     output_actions=[Output.BNC1High, Output.BNC2Low],
        # )

        # self.bpod.add_state(
        #     state_name="fifth",
        #     state_timer=3,
        #     state_change_conditions={Event.Tup: "exit"},
        #     #output_actions=[Output.SoftCode2, Output.BNC1High],
        #     output_actions=[Output.BNC1Low, Output.BNC2Low],
        # )

        # # 'ready_to_initiate': state that turns on the central port light and
        # # waits for a poke in the central port (Port2)
        # self.bpod.add_state(
        #     state_name="ready_to_initiate",
        #     state_timer=0,
        #     state_change_conditions={Event.Port2In: "stimulus_state"},
        #     output_actions=[(Output.PWM2, self.settings.light_intensity_high)],
        # )

        # # 'stimulus_state': state that turns on the side ports and
        # # waits for a poke in one of the side ports (Port1 or Port3)
        # self.bpod.add_state(
        #     state_name="stimulus_state",
        #     state_timer=0,
        #     state_change_conditions={
        #         Event.Port1In: "reward_state_left",
        #         Event.Port3In: "reward_state_right",
        #     },
        #     output_actions=[
        #         (Output.PWM1, self.settings.light_intensity_high),
        #         (Output.PWM3, self.settings.light_intensity_high),
        #     ],
        # )

        # # 'reward_state_left' and 'reward_state_right': states that deliver the reward
        # self.bpod.add_state(
        #     state_name="reward_state_left",
        #     state_timer=self.left_valve_opening_time,
        #     state_change_conditions={Event.Tup: "exit"},
        #     output_actions=[Output.Valve1],
        # )

        # self.bpod.add_state(
        #     state_name="reward_state_right",
        #     state_timer=self.right_valve_opening_time,
        #     state_change_conditions={Event.Tup: "exit"},
        #     output_actions=[Output.Valve3],
        # )

    def after_trial(self):
        """
        Here you can register all the values you need to save for each trial.
        It is essential to always include a variable named water, which stores the
        amount of water consumed during each trial.
        The system will calculate the total water consumption in each session
        by summing this variable.
        If the total water consumption falls below a certain threshold,
        an alarm will be triggered.
        This threshold can be adjusted in the Settings tab of the GUI.
        """

        pass

        # self.register_value("water", self.settings.reward_amount_ml)

    def close(self):
        """
        Here you can perform any actions you want to take once the task is completed,
        such as sending a message via email or Slack, creating a plot, and more.
        """

        pass
