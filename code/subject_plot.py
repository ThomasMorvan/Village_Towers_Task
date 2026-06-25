import calplot
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from village.custom_classes.subject_plot_base import SubjectPlotBase
from plot_utils import plot_stage_progression, plot_difficulty_progression


class SubjectPlot(SubjectPlotBase):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                    width: float = 10, height: float = 12) -> Figure:

        dates_df = df.date.value_counts(sort=False)
        dates_df.index = pd.to_datetime(dates_df.index)

        cpfig, _ = calplot.calplot(data=dates_df)
        canvas = FigureCanvasAgg(cpfig)
        canvas.draw()
        cal_w, cal_h = cpfig.get_size_inches() * cpfig.get_dpi()
        image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(
            int(cal_h), int(cal_w), 4)
        plt.close(cpfig)

        has_stage = ("phase" in df.columns
                     and "stage" in df.columns
                     and "session" in df.columns)
        has_difficulty = (has_stage and
                          ("mu_nr" in df.columns or "led_ms" in df.columns))

        n_rows = 2 + int(has_stage) + int(has_difficulty)
        fig, axs = plt.subplots(n_rows, 1, figsize=(width, height))

        axs[0].imshow(image)
        axs[0].axis("off")

        axs[1].bar(dates_df.index, dates_df.values, width=0.8)
        loc = mdates.AutoDateLocator(maxticks=12)
        axs[1].xaxis.set_major_locator(loc)
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        axs[1].set_ylabel("Number of trials")
        axs[1].set_xlabel("Date")

        if has_stage:
            plot_stage_progression(df, axs[2])
        if has_difficulty:
            plot_difficulty_progression(df, axs[3 if has_stage else 2])

        fig.tight_layout()
        return fig
