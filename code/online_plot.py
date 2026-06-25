import pandas as pd
from matplotlib import pyplot as plt
from village.custom_classes.online_plot_base import OnlinePlotBase
from village.manager import manager
from task_stages import REQUIRED_COLS
from plot_utils import (shade_phases, shade_stages, mark_checkpoints,
                        plot_staircase, plot_rolling_accuracy,
                        plot_streak, plot_step, shade_rescue, to_time_axis)


class Online_Plot(OnlinePlotBase):
    def __init__(self) -> None:
        super().__init__()

    def create_figure_and_axes(self, width=14, height=8):
        self.fig, axs = plt.subplots(2, 2, figsize=(width, height),
                                     layout="constrained")
        (self.ax1, self.ax2), (self.ax3, self.ax4) = axs
        self._staircase_twin = self.ax1.twinx()
        self._step_twin = self.ax4.twinx()
        dpi = self.fig.get_dpi()
        self.window_geometry = (100, 50,
                                int(width * dpi), int(height * dpi))

    @staticmethod
    def _clear_ax(ax) -> None:
        """Custom clear for axes that have a twin because cla() also clears
        the twin, which we don't want."""
        for coll in (ax.lines, ax.collections, ax.patches,
                     ax.texts, ax.images, ax.containers):
            for a in list(coll):
                try:
                    a.remove()
                except Exception:
                    pass
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.relim()

    @staticmethod
    def _clear_twin(twin_ax=None):
        """Remove data artists from twin without calling cla()."""
        if twin_ax is None:
            return

        for artist_list in (twin_ax.lines, twin_ax.collections,
                            twin_ax.patches, twin_ax.texts,
                            twin_ax.containers):
            for a in list(artist_list):
                try:
                    a.remove()
                except Exception:
                    pass
        leg = twin_ax.get_legend()
        if leg is not None:
            leg.remove()
        twin_ax.yaxis.tick_right()
        twin_ax.yaxis.set_label_position("right")

    def update_plot(self, df: pd.DataFrame) -> None:
        if df.empty or not REQUIRED_COLS.issubset(df.columns):
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                self._error_plot(ax, "Waiting for data...")
            return

        # Staircase/accuracy spaced by time; streak/step keep the trial index.
        df_t, xlabel = to_time_axis(df)
        for ax, fn, d, xl in ((self.ax1, self._plot_staircase, df_t, xlabel),
                              (self.ax2, self._plot_rolling_accuracy,
                               df_t, xlabel),
                              (self.ax3, self._plot_streak, df, "Trial"),
                              (self.ax4, self._plot_step, df, "Trial")):
            try:
                fn(d, ax)
                ax.set_xlabel(xl)
            except Exception as e:
                self._error_plot(ax, str(e))

    def _plot_staircase(self, df, ax):
        self._clear_ax(ax)
        self._clear_twin(self._staircase_twin)
        if df.empty:
            self._error_plot(ax, "No data yet")
            return
        shade_stages(ax, df)
        mark_checkpoints(ax, df)
        shade_phases(ax, df)
        plot_staircase(df, ax, twin_ax=self._staircase_twin)

    def _plot_rolling_accuracy(self, df, ax):
        ax.clear()
        shade_stages(ax, df)
        mark_checkpoints(ax, df)
        shade_phases(ax, df)
        shade_rescue(ax, df)
        s = getattr(manager.training, "settings", None)
        window = int(getattr(s, "acc_window", 40))
        plot_rolling_accuracy(df, ax, window=window)

    def _plot_streak(self, df, ax):
        ax.clear()
        plot_streak(df, ax)

    def _plot_step(self, df, ax):
        self._clear_ax(ax)
        self._clear_twin(self._step_twin)
        shade_stages(ax, df)
        shade_phases(ax, df)
        plot_step(df, ax, twin_ax=self._step_twin)

    def _error_plot(self, ax, msg="Could not create plot"):
        self._clear_ax(ax)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes)
