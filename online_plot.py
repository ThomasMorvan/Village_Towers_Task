import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from village.custom_classes.online_plot_base import OnlinePlotBase


class Online_Plot(OnlinePlotBase):
    def __init__(self) -> None:
        super().__init__()

    def create_figure_and_axes(self, width=10, height=8):
        self.fig = plt.figure(figsize=(width, height))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

    def update_plot(self, df: pd.DataFrame) -> None:
        try:
            self.make_timing_plot(df, self.ax1)
        except Exception:
            self.make_error_plot(self.ax1)
        try:
            self.make_trial_side_and_correct_plot(df, self.ax2)
        except Exception:
            self.make_error_plot(self.ax2)

        self.fig.tight_layout()

    def make_timing_plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        ax.clear()
        df.plot(kind="scatter", x="TRIAL_START", y="trial", ax=ax)

    def make_trial_side_and_correct_plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        _ = self.plot_side_correct_performance(df, ax)

    def make_error_plot(self, ax) -> None:
        ax.clear()
        ax.text(
            0.5,
            0.5,
            "Could not create plot",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    def plot_side_correct_performance(df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
        ax.clear()
        # select only the last 100 trials
        df = df.tail(100)
        sns.scatterplot(data=df, x="trial", y="trial_type", hue="correct", ax=ax)
        # make sure the y axis ticks are ascending, inverting the y axis
        ax.invert_yaxis()
        # plot the mean of the last 10 trials
        ax.plot(pd.Series([int(x) for x in df.correct]).rolling(10).mean(), "r")

        return ax
