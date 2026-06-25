import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from village.custom_classes.session_plot_base import SessionPlotBase
from task_stages import REQUIRED_COLS
from plot_utils import (shade_phases, shade_stages, mark_checkpoints,
                        plot_staircase, plot_rolling_accuracy,
                        plot_psychometric, plot_lr_bars, plot_stage_diagnostic,
                        shade_rescue, to_time_axis)


class SessionPlot(SessionPlotBase):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Session Plot"

    SESSION_REQUIRED_COLS = REQUIRED_COLS | {"mu_r",
                                             "delta_towers",
                                             "checkpoint",
                                             "checkpoint_floor"}

    def create_plot(self, df: pd.DataFrame, weight: float = 0,
                    width: float = 10, height: float = 10) -> Figure:
        if df.empty or not self.SESSION_REQUIRED_COLS.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(width, height))
            ax.text(0.5, 0.5, "Waiting for data...",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        # Space trial-axis plots by real time within the session.
        df, self._xlabel = to_time_axis(df)

        # Last row split: L/R bars (left half) + advance diagnostic (right).
        fig, axd = plt.subplot_mosaic(
            [["stair", "stair"],
             ["acc", "acc"],
             ["psy", "psy"],
             ["lr", "diag"]],
            figsize=(width, height),
            gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5]})
        ax1, ax2 = axd["stair"], axd["acc"]
        for ax, fn in ((axd["stair"], self._plot_staircase),
                       (axd["acc"], self._plot_rolling_accuracy),
                       (axd["psy"], self._plot_psychometric),
                       (axd["lr"], self._plot_lr_bars),
                       (axd["diag"], self._plot_stage_diagnostic)):
            try:
                fn(df, ax)
            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=7,
                        color="red")
        ax1.set_xlabel(self._xlabel)
        ax2.set_xlabel(self._xlabel)
        fig.tight_layout()
        return fig

    def _plot_staircase(self, df, ax):
        ax.clear()
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return
        shade_stages(ax, df)
        mark_checkpoints(ax, df)
        shade_phases(ax, df)
        plot_staircase(df, ax)

    def _plot_rolling_accuracy(self, df, ax):
        ax.clear()
        shade_stages(ax, df)
        mark_checkpoints(ax, df)
        shade_phases(ax, df)
        shade_rescue(ax, df)
        s = getattr(self, "settings", None)
        window = int(getattr(s, "acc_window", 40))
        plot_rolling_accuracy(df, ax, window=window)

    def _plot_psychometric(self, df, ax):
        ax.clear()
        plot_psychometric(df, ax)

    def _plot_lr_bars(self, df, ax):
        ax.clear()
        plot_lr_bars(df, ax)

    def _plot_stage_diagnostic(self, df, ax):
        ax.clear()
        s = getattr(self, "settings", None)
        window = int(getattr(s, "acc_window", 40))
        plot_stage_diagnostic(df, ax, window=window)
