import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from village.custom_classes.session_plot_base import SessionPlotBase
from task_stages import REQUIRED_COLS
from plot_utils import (shade_phases, shade_stages, mark_checkpoints,
                        plot_staircase, plot_rolling_accuracy,
                        plot_psychometric, shade_rescue)


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

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width, height),
                                            gridspec_kw={"height_ratios":
                                                         [2, 2, 1.5]})
        for ax, fn in ((ax1, self._plot_staircase),
                       (ax2, self._plot_rolling_accuracy),
                       (ax3, self._plot_psychometric)):
            try:
                fn(df, ax)
            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=7,
                        color="red")
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
        plot_rolling_accuracy(df, ax, window=int(self.settings.acc_window),
                              rescue_threshold=self.settings.rescue_threshold)

    def _plot_psychometric(self, df, ax):
        ax.clear()
        plot_psychometric(df, ax)
