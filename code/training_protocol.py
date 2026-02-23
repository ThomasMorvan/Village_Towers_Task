from village.custom_classes.training_protocol_base import TrainingProtocolBase


# click on the link below to see the documentation about how to create
# tasks, plots and training protocols
# https://braincircuitsbehaviorlab.github.io/village/user_guide/create.html


class TrainingProtocol(TrainingProtocolBase):
    """
    This class defines the training protocol for animal behavior experiments.
    The training protocol is run every time a task is finished and it determines:
    1. Which new task is scheduled for the subject
    2. How training variables change based on performance metrics

    Required methods to implement:
    - __init__: Initialize the training protocol
    - default_training_settings: Define initial parameters. It is called when creating a new subject.
    - update_training_settings: Update parameters after each session.

    Optional method:
    - gui_tabs: Organize the variables in custom GUI tabs
    """

    def __init__(self) -> None:
        """Initialize the training protocol."""
        super().__init__()

    def default_training_settings(self) -> None:
        """
        Define all initial training parameters for new subjects.

        This method is called when creating a new subject, and these parameters
        are saved as the initial values for that subject.

        Required parameters:
        - next_task (str): Name of the next task to run
        - refractary_period (int): Waiting time in seconds between sessions
        - minimum_duration (int): Minimum time in seconds for the task before door2 opens
        - maximum_duration (int): Maximum time in seconds before task stops automatically

        Additional parameters:
        You can define any additional parameters needed for your specific tasks.
        These can be modified between sessions based on subject performance.
        """

        # Required parameters for any training protocol
        self.settings.next_task = "Habituation"  # Next task to run
        self.settings.refractory_period = (
            3600 * 4
        )  # 4 hours between sessions of the same subject
        self.settings.minimum_duration = 600  # Minimum duration of 10 min
        self.settings.maximum_duration = 900  # Maximum duration of 15 min

        # Task-specific parameters
        # (can be modified between sessions or set when the task is run manually)
        self.settings.reward_amount_ml = 0.08  # Reward volume in milliliters
        self.settings.stage = 1  # Current training stage
        self.settings.light_intensity_high = (
            255  # High light intensity in the port (0-255)
        )
        self.settings.light_intensity_low = (
            50  # Low light intensity in the port (0-255)
        )
        self.settings.trial_types = "left_easy"
        self.settings.punishment_time = 1  # Time in seconds for punishment
        self.settings.iti_time = 2  # Inter-trial interval in seconds
        self.settings.response_time = 10  # Time in seconds to respond before timeout

        self.settings.size = 200
        self.settings.image_png = "image.png"
        self.settings.image_jpg = "image.jpg"
        self.settings.video = "video1.avi"
        self.settings.stimulus_duration = 1
        self.settings.color = "cyan"
        self.settings.x_position = 100
        self.settings.y_position = 300
        self.settings.width = 500
        self.settings.height = 30
        self.settings.background_color = (30, 30, 30)

    def update_training_settings(self) -> None:
        """
        Update training parameters after each session.

        This method is called when a session finishes and determines how
        the subject progresses through the training protocol.

        Available data for decision-making:
        - self.subject (str): Name of the current subject
        - self.last_task (str): Name of the task that just finished
        - self.df (pd.DataFrame): DataFrame with all sessions data for this subject

        Example logic:
        - Progress from Habituation to FollowTheLight after 2 sessions with >100 trials
        - Reduce reward amount as training progresses
        - Advance to stage 2 after two consecutive sessions in FollowTheLight with >85% performance
        """

        if self.last_task == "Habituation":
            # Get all Habituation sessions from the dataframe
            df_habituation = self.df[self.df["task"] == "Habituation"]

            # Check if the animal completed at least 2 Habituation sessions
            if len(df_habituation) >= 2:
                # Get data from the last session
                df_last_session = df_habituation.iloc[-1]
                trials_last_session = df_last_session["trial"].iloc[-1]

                # Progress to next task if criteria met (>100 trials)
                if trials_last_session >= 100:
                    self.settings.next_task = "FollowTheLight"
                    self.settings.reward_amount_ml = 0.07  # Decrease reward

        elif self.last_task == "FollowTheLight":
            # Get all FollowTheLight sessions
            df_follow_the_light = self.df[self.df["task"] == "FollowTheLight"]

            # Check if at least 2 sessions completed
            if len(df_follow_the_light) >= 2:
                # Get data from the last two sessions
                df_last_session = df_follow_the_light.iloc[-1]
                df_previous_session = df_follow_the_light.iloc[-2]

                # Calculate performance metrics
                performance_last_session = df_last_session["correct"].mean()
                performance_previous_session = df_previous_session["correct"].mean()
                trials_last_session = df_last_session["trial"].iloc[-1]
                trials_previous_session = df_previous_session["trial"].iloc[-1]

                # Advance to stage 2 if criteria met
                # (>85% correct in two sessions with >100 trials each)
                if (
                    performance_last_session >= 0.85
                    and performance_previous_session >= 0.85
                    and trials_last_session >= 100
                    and trials_previous_session >= 100
                ):
                    self.settings.stage = 2
                    self.settings.reward_amount_ml = 0.05  # Decrease reward

    def define_gui_tabs(self):
        """
        Define the organization of the settings in the GUI.

        Whatever that is not defined here will be placed in the "General" tab.
        They need to have the same name as your settings variables.
        You can use the 'Hide' tab to hide a setting from the GUI.
        Items in the lists need to have the same name as your settings variables.
        You can also restrict the possible values for each setting.
        """
        self.gui_tabs = {
            "Port_variables": [
                "reward_amount_ml",
                "light_intensity_high",
                "light_intensity_low",
            ],
            "Other_variables": [
                "stage",
                "trial_types",
                "punishment_time",
                "iti_time",
                "response_time",
            ],
        }

        # Define possible values for each variable
        self.gui_tabs_restricted = {
            "trial_types": ["left_easy", "right_easy", "left_hard", "right_hard"],
        }
