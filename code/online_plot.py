import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import seaborn as sns
from village.custom_classes.online_plot_base import OnlinePlotBase

# Related to test_temp. plot the temperature of the live plot.


class Online_Plot(OnlinePlotBase):
    def __init__(self) -> None:
        super().__init__()

    def create_figure_and_axes(self, width=10, height=8):
        print("MAKING NEW FIGURE")
        self.fig = plt.figure(figsize=(width, height))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

    def update_plot(self, df: pd.DataFrame) -> None:
        try:
            self.make_timing_plot(df, self.ax1)
        except Exception:
            self.make_error_plot(self.ax1)
        try:
            self.make_trial_side_and_correct_plot(df, self.ax2)
        except Exception:
            self.make_error_plot(self.ax2)
        try:
            self.make_temperature_plot(df, self.ax3)
        except Exception:
            self.make_error_plot(self.ax3)

        self.fig.tight_layout()

    @staticmethod
    def smooth(x, y):
        x = np.array(x)
        y = np.array(y)
        f = interp1d(x, y, kind="quadratic")

        x_new = np.linspace(min(x), max(x), 100)
        y_new = f(x_new)
        return x, y

    def make_temperature_plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        ax.clear()
        x = df["TRIAL_START"]
        y = df["temperature"]
        ax.plot(x, y, ms=7.5, lw=0, marker='o', mew=1.5, mec="g", mfc="w")
        new_x, new_y = self.smooth(x, y)
        ax.plot(new_x, new_y, ms=0, lw=1.5, marker='o', color="darkgray")
        ax.axhline(y=30, lw=1, color="lightgray", ls="--")
        # df.plot(kind="scatter", x="TRIAL_START", y="temperature", ax=ax)

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
