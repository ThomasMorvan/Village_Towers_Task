import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from task_stages import STAGES, CHECKPOINT_COLORS

CFG = {
    # shading
    "warmup_color":      "gold",
    "warmup_alpha":      0.15,
    "stage_alpha":       0.18,
    "session_alpha":     0.25,

    # staircase lines
    "mu_r_color":        "dodgerblue",
    "mu_r_lw":           1.2,
    "mu_r_alpha":        0.7,
    "mu_nr_color":       "orange",
    "mu_nr_lw":          1.5,
    "led_color":         "cyan",
    "led_lw":            1.5,
    "delta_color":       "seagreen",
    "delta_alpha":       0.35,
    "delta_ms":          10,

    # targets and floor
    "target_ls":         "--",
    "target_lw":         1.0,
    "target_alpha":      0.6,
    "floor_color":       "gray",
    "floor_ls":          ":",
    "floor_lw":          1.0,
    "floor_alpha":       0.8,

    # rolling accuracy
    "acc_color":         "navy",
    "acc_lw":            1.5,
    "thr_s1_color":      "green",
    "thr_s23_color":     "orange",
    "thr_lw":            1.0,
    "thr_alpha":         0.7,
    "acc_ylim":          (-0.05, 1.1),

    # streak
    "streak_ok":         "forestgreen",
    "streak_err":        "red",
    "streak_alpha":      0.5,
    "streak_lw":         0.8,
    "streak_line_alpha": 0.6,

    # checkpoints
    "ck_lw":             1.5,
    "ck_alpha":          0.8,

    # rescue
    "rescue_color":      "indianred",
    "rescue_alpha":      0.20,
    "rescue_thr_ls":     "--",
    "rescue_thr_lw":     1.0,
    "rescue_thr_alpha":  0.8,

    # psychometric
    "psycho_lw":         1.5,
    "psycho_ms":         5,

    # subject-level plots
    "subj_ms":           5,
    "subj_lw":           1.5,

    # text / labels
    "fs":                7,
    "fs_label":          8,
}


def shade_phases(ax, df):
    """warmup phase shade."""
    if "phase" not in df.columns:
        return
    df = df.reset_index(drop=True)
    trials = df["trial"].tolist()
    phases = df["phase"].tolist()
    start, cur = trials[0], phases[0]
    for t, p in zip(trials[1:], phases[1:]):
        if p != cur:
            if cur == "warmup":
                ax.axvspan(start - 0.5, t - 0.5,
                           alpha=CFG["warmup_alpha"],
                           color=CFG["warmup_color"], zorder=0,
                           lw=0)
            start, cur = t, p
    if cur == "warmup":
        ax.axvspan(start - 0.5, trials[-1] + 0.5,
                   alpha=CFG["warmup_alpha"],
                   color=CFG["warmup_color"], zorder=0,
                   lw=0)


def shade_rescue(ax, df):
    """Shade rescue blocks in red."""
    if "rescue" not in df.columns:
        return
    trials = df["trial"].tolist()
    rescues = df["rescue"].fillna(0).astype(int).tolist()
    in_block = False
    for t, r in zip(trials, rescues):
        if r and not in_block:
            start = t
            in_block = True
        elif not r and in_block:
            ax.axvspan(start - 0.5, t - 0.5,
                       color=CFG["rescue_color"], alpha=CFG["rescue_alpha"],
                       zorder=1, lw=0, label="Rescue")
            in_block = False
    if in_block:
        ax.axvspan(start - 0.5, trials[-1] + 0.5,
                   color=CFG["rescue_color"], alpha=CFG["rescue_alpha"],
                   zorder=1, lw=0, label="Rescue")


def shade_stages(ax, df):
    """Background color for each stage."""
    df = df.reset_index(drop=True)
    cur_s = df["stage"].iloc[0]
    start_t = df["trial"].iloc[0]
    for _, row in df.iterrows():
        if row["stage"] != cur_s:
            cfg = STAGES.get(int(cur_s))
            ax.axvspan(start_t - 0.5, row["trial"] - 0.5,
                       alpha=CFG["stage_alpha"],
                       color=cfg.color if cfg else "w", zorder=0, lw=0)
            ax.text((start_t + row["trial"]) / 2, 1.02, f"S{int(cur_s)}",
                    ha="center", fontsize=CFG["fs"],
                    transform=ax.get_xaxis_transform())
            start_t, cur_s = row["trial"], row["stage"]
    cfg = STAGES.get(int(cur_s))
    ax.axvspan(start_t - 0.5, df["trial"].iloc[-1] + 0.5,
               alpha=CFG["stage_alpha"],
               color=cfg.color if cfg else "w", zorder=0, lw=0)


def mark_checkpoints(ax, df, y_frac=0.95):
    """Vertical dashed lines where the checkpoint index increases."""
    if "checkpoint" not in df.columns:
        return
    cp = df["checkpoint"].fillna(0).astype(int)
    for _, row in df[cp > cp.shift(fill_value=0)].iterrows():
        gate = int(row["checkpoint"])
        color = CHECKPOINT_COLORS[min(gate - 1, len(CHECKPOINT_COLORS) - 1)]
        ax.axvline(row["trial"], color=color,
                   lw=CFG["ck_lw"], ls="--", alpha=CFG["ck_alpha"])
        ax.text(row["trial"], y_frac, f"Chck{gate}", ha="center",
                fontsize=CFG["fs"], color=color,
                transform=ax.get_xaxis_transform())


def plot_staircase(df, ax, twin_ax=None):
    """Plot difficulty mu_r, mu_nr (left axis) and led_ms (right axis)."""
    ax2 = twin_ax if twin_ax is not None else ax.twinx()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    df_mu_r = df.dropna(subset=["mu_r"])
    if not df_mu_r.empty:
        ax.plot(df_mu_r["trial"], df_mu_r["mu_r"],
                color=CFG["mu_r_color"], lw=CFG["mu_r_lw"],
                alpha=CFG["mu_r_alpha"], label=r"$\mu_{R}$")

    df_mu_nr = df.dropna(subset=["mu_nr"])
    if not df_mu_nr.empty:
        ax.plot(df_mu_nr["trial"], df_mu_nr["mu_nr"],
                color=CFG["mu_nr_color"], lw=CFG["mu_nr_lw"],
                label=r"$\mu_{NR}$")
        ax.axhline(STAGES[2].staircase.target,
                   color=CFG["mu_nr_color"],
                   ls=CFG["target_ls"], lw=CFG["target_lw"],
                   alpha=CFG["target_alpha"],
                   label=f"S2 target {STAGES[2].staircase.target}")
        ax.axhline(STAGES[4].staircase.target,
                   color=CFG["mu_nr_color"],
                   ls=":", lw=CFG["target_lw"],
                   alpha=CFG["target_alpha"],
                   label=f"S4 target {STAGES[4].staircase.target}")

    if "delta_towers" in df.columns:
        df_dt = df.dropna(subset=["delta_towers"])
        if not df_dt.empty:
            ax.scatter(df_dt["trial"], df_dt["delta_towers"],
                       s=CFG["delta_ms"], color=CFG["delta_color"],
                       alpha=CFG["delta_alpha"], edgecolors="none",
                       zorder=1, label=r"$\Delta$towers")

    df_led = df.dropna(subset=["led_ms"])
    if not df_led.empty:
        ax2.plot(df_led["trial"], df_led["led_ms"],
                 color=CFG["led_color"], lw=CFG["led_lw"], label="led_ms")
        ax2.axhline(STAGES[3].staircase.target,
                    color=CFG["led_color"],
                    ls=CFG["target_ls"], lw=CFG["target_lw"],
                    alpha=CFG["target_alpha"],
                    label=f"led_ms target {STAGES[3].staircase.target:.0f}")

    if "checkpoint_floor" in df.columns:
        floor = df["checkpoint_floor"].dropna()
        if len(floor) and floor.iloc[-1] > 0:
            cur_stage = (int(df["stage"].iloc[-1]) if "stage" in df.columns
                         else 0)
            if cur_stage == 3:
                ax2.axhline(floor.iloc[-1], color=CFG["floor_color"],
                            ls=CFG["floor_ls"], lw=CFG["floor_lw"],
                            alpha=CFG["floor_alpha"])
            else:
                ax.axhline(floor.iloc[-1], color=CFG["floor_color"],
                           ls=CFG["floor_ls"], lw=CFG["floor_lw"],
                           alpha=CFG["floor_alpha"],
                           label=f"floor {floor.iloc[-1]:.2f}")

    if "stage" in df.columns and len(df):
        cur_stage = int(df["stage"].iloc[-1])
        cfg = STAGES.get(cur_stage)
        ax.text(0.02, 0.95, f"S{cur_stage}: {cfg.name if cfg else '?'}",
                fontsize=CFG["fs_label"], transform=ax.transAxes, va="top",
                color=cfg.color if cfg else "black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_ylabel("Tower count", color=CFG["mu_nr_color"])
    ax.set_xlabel("Trial")
    top, bottom = 9.0, 0.0
    if "delta_towers" in df.columns and df["delta_towers"].notna().any():
        dt = df["delta_towers"]
        top = max(top, float(dt.max()) + 1)
        bottom = min(bottom, float(dt.min()) - 1)
    ax.set_ylim(bottom, top)
    ax2.set_ylabel("led_ms (ms)", color=CFG["led_color"])
    ax.tick_params(axis="y", labelcolor=CFG["mu_nr_color"])
    ax2.tick_params(axis="y", labelcolor=CFG["led_color"])
    ax2.set_yticks(np.arange(0, 5100, 1000))
    ax2.set_ylim(0, 5100)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=CFG["fs"], loc="center left", ncols=3)

    return ax2


def plot_rolling_accuracy(df, ax, window: int = 100,
                          rescue_threshold: float | None = None):
    df = df.dropna(subset=["trial_correct"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No trial data", ha="center", va="center",
                transform=ax.transAxes)
        return

    rolling = df["trial_correct"].astype(float).rolling(window, min_periods=1)
    rolling = rolling.mean()

    ok = df["trial_correct"].astype(bool)
    ax.vlines(df.loc[ok, "trial"], 0.96, 1.0,
              color=CFG["streak_ok"], alpha=0.5, lw=0.6, zorder=1)
    ax.vlines(df.loc[~ok, "trial"], 1.0, 1.04,
              color=CFG["streak_err"], alpha=0.5, lw=0.6, zorder=1)
    ax.plot([], [], color=CFG["streak_ok"], marker="|", lw=0,
            markersize=7, alpha=0.8, label="Correct trials")
    ax.plot([], [], color=CFG["streak_err"], marker="|", lw=0,
            markersize=7, alpha=0.8, label="Incorrect trials")

    ax.plot(df["trial"], rolling, color=CFG["acc_color"], lw=CFG["acc_lw"],
            label=f"Rolling accuracy ({window} trials)", zorder=2)

    ax.axhline(STAGES[1].advance_threshold, color=CFG["thr_s1_color"],
               ls=CFG["target_ls"], lw=CFG["thr_lw"], alpha=CFG["thr_alpha"],
               label=f"S1 advance ({STAGES[1].advance_threshold})")
    ax.axhline(STAGES[2].advance_threshold, color=CFG["thr_s23_color"],
               ls=CFG["target_ls"], lw=CFG["thr_lw"], alpha=CFG["thr_alpha"],
               label=f"S2-4 advance ({STAGES[2].advance_threshold})")
    if rescue_threshold is not None:
        ax.axhline(rescue_threshold, color=CFG["rescue_color"],
                   ls=CFG["rescue_thr_ls"], lw=CFG["rescue_thr_lw"],
                   alpha=CFG["rescue_thr_alpha"],
                   label=f"Rescue ({rescue_threshold:.0%})")

    if "stage" in df.columns:
        for _, row in df[df["stage"] != df["stage"].shift()].iterrows():
            ax.text(row["trial"], 1.02, f"S{int(row['stage'])}",
                    ha="center", fontsize=CFG["fs"],
                    transform=ax.get_xaxis_transform())

    ax.set_ylim(*CFG["acc_ylim"])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Trial")
    ax.legend(fontsize=CFG["fs"], loc="lower left", ncols=2)


def _insert_zeros(x, y):
    """Insert points where y crosses zero, to make
    the streaks look better."""
    out_x, out_y = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i - 1] * y[i] < 0:
            t = x[i-1] + (-y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
            out_x.append(t)
            out_y.append(0.0)
        out_x.append(x[i])
        out_y.append(y[i])
    return np.array(out_x), np.array(out_y)


def plot_streak(df, ax):
    """Streak history: positive = correct streak, negative = error streak."""
    df = df.dropna(subset=["streak"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No streak data", ha="center", va="center",
                transform=ax.transAxes)
        return
    if (df["streak"] == 0).all():
        ax.text(0.5, 0.5, "Streak not tracked\n(no staircase in this stage)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    trials = df["trial"].values
    streak = df["streak"].astype(float).values
    xi, yi = _insert_zeros(trials, streak)

    ax.fill_between(xi, yi, 0, where=(yi >= 0),
                    color=CFG["streak_ok"], alpha=CFG["streak_alpha"],
                    label="Correct streak")
    ax.fill_between(xi, yi, 0, where=(yi <= 0),
                    color=CFG["streak_err"], alpha=CFG["streak_alpha"],
                    label="Incorrect streak")
    ax.plot(xi, yi, color="black",
            lw=CFG["streak_lw"], alpha=CFG["streak_line_alpha"])
    ax.axhline(0, color="black", lw=CFG["streak_lw"])
    ax.set_ylabel("Streak")
    ax.set_xlabel("Trial")
    ax.legend(fontsize=CFG["fs"], loc="upper left")


def _step_bars(ax, sub_df, boost_series, ok_color, err_color):
    """Draw the boost as stacked step bars:
        base (correct/incorrect) + gold boost top."""
    if sub_df.empty:
        return
    boost = boost_series.reindex(sub_df.index).fillna(1.0)
    base = sub_df["step_delta"] / boost
    extra = sub_df["step_delta"] - base
    ok = sub_df["trial_correct"].astype(bool)
    ax.bar(sub_df.loc[ok,  "trial"], base[ok],
           color=ok_color,  alpha=CFG["streak_alpha"],
           width=1, label="correct")
    ax.bar(sub_df.loc[~ok, "trial"], base[~ok],
           color=err_color, alpha=CFG["streak_alpha"],
           width=1, label="incorrect")
    boosted = extra > 1e-9
    if boosted.any():
        ax.bar(sub_df.loc[boosted, "trial"], extra[boosted],
               bottom=base[boosted], color="gold", alpha=0.7, width=1,
               label="boost")


def plot_step(df, ax, twin_ax=None):
    """Step size per trial.
    Left axis: density steps (S2/S4). Right axis: ms steps (S3, ms).
    Bars split into base step + gold stacked top for boost contribution.
    Twin axes to avoid scale clash."""
    if "step_delta" not in df.columns:
        ax.text(0.5, 0.5, "No step_delta column", ha="center", va="center",
                transform=ax.transAxes)
        return twin_ax
    df = df[df["step_delta"] > 0].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No staircase steps yet", ha="center", va="center",
                transform=ax.transAxes)
        return twin_ax

    ax2 = twin_ax if twin_ax is not None else ax.twinx()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    boost = df["step_boost"] if "step_boost" in df.columns else None
    df_dens = df[df["stage"].isin([2, 4])] if "stage" in df.columns else df
    _step_bars(ax, df_dens,
               boost if boost is not None else df_dens["step_delta"],
               CFG["streak_ok"], CFG["streak_err"])

    df_ms = df[df["stage"] == 3] if "stage" in df.columns else pd.DataFrame()
    _step_bars(ax2, df_ms,
               (boost if boost is not None
                else df_ms["step_delta"] if not df_ms.empty
                else pd.Series(dtype=float)),
               CFG["led_color"], "darkorange")

    ax.set_ylabel("Step size", color=CFG["mu_nr_color"])
    ax2.set_ylabel("Δ led_ms (ms)", color=CFG["led_color"])
    ax.tick_params(axis="y", labelcolor=CFG["mu_nr_color"])
    ax2.tick_params(axis="y", labelcolor=CFG["led_color"])
    ax.set_xlabel("Trial")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    seen, lines, labels = set(), [], []
    for line, lbl in zip(lines1 + lines2, labels1 + labels2):
        if lbl not in seen:
            seen.add(lbl)
            lines.append(line)
            labels.append(lbl)
    ax.legend(lines, labels, fontsize=CFG["fs"], loc="upper left")
    return ax2


def plot_psychometric(df, ax):
    """Psychometric curve per stage, main phase only."""
    df_main = df[(df["phase"] == "main") & (df["stage"] >= 1)]
    df_main = df_main.dropna(subset=["delta_towers", "trial_correct"]).copy()

    if df_main.empty:
        ax.text(0.5, 0.5, "No main-phase data (stages 1--3)",
                ha="center", va="center", transform=ax.transAxes)
        return

    stages = sorted(df_main["stage"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, max(len(stages), 1)))

    for stage, color in zip(stages, colors):
        df_s = df_main[df_main["stage"] == stage].copy()
        if len(df_s) < 5:
            continue
        bins = np.arange(df_s["delta_towers"].min() - 0.5,
                         df_s["delta_towers"].max() + 1.5, 1)
        df_s["bin"] = pd.cut(df_s["delta_towers"], bins=bins,
                             labels=False, include_lowest=True)
        grouped = df_s.groupby("bin").agg(x=("delta_towers", "mean"),
                                          y=("trial_correct", "mean")).dropna()
        if len(grouped) >= 2:
            ax.plot(grouped["x"], grouped["y"], "o-", color=color,
                    label=f"Stage {int(stage)}",
                    lw=CFG["psycho_lw"], ms=CFG["psycho_ms"])

    ax.axhline(0.5, color=CFG["floor_color"],
               ls=CFG["target_ls"], lw=CFG["target_lw"])
    ax.set_xlabel("Δ Towers (r - nr)")
    ax.set_ylabel("Fraction correct")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=CFG["fs"], loc="upper left")


def plot_stage_progression(df, ax):
    """Start/end stage per session."""
    df_plot = df[df["phase"] == "main"] if "phase" in df.columns else df
    if df_plot.empty:
        ax.text(0.5, 0.5, "No main-phase data",
                ha="center", va="center", transform=ax.transAxes)
        return

    stage_info = df_plot.groupby("session")["stage"].agg(["first", "last"])
    stage_info = stage_info.reset_index()
    for _, row in stage_info.iterrows():
        cfg = STAGES.get(int(row["last"]))
        ax.axvspan(row["session"] - 0.5, row["session"] + 0.5,
                   alpha=CFG["session_alpha"], zorder=0, lw=0,
                   color=cfg.color if cfg else "lightgray")

    ax.plot(stage_info["session"], stage_info["first"], "o-",
            color="steelblue", label="Start stage", ms=CFG["subj_ms"])
    ax.plot(stage_info["session"], stage_info["last"], "s--",
            color="darkorange", label="End stage", ms=CFG["subj_ms"])
    ax.set_yticks(range(0, len(STAGES)))
    ax.set_yticklabels([f"{s}] {STAGES[s].name}" for s in range(len(STAGES))])
    ax.set_ylabel("Stage")
    ax.set_xlabel("Session")
    ax.legend(fontsize=CFG["fs_label"], loc="upper left")


def plot_difficulty_progression(df, ax):
    """Median mu_nr (S2, left axis) and led_ms (S3, right axis) per session."""
    df_main = (df[df["phase"] == "main"].copy() if "phase" in df.columns
               else df.copy())
    if df_main.empty:
        ax.text(0.5, 0.5, "No difficulty data",
                ha="center", va="center", transform=ax.transAxes)
        return

    ax2 = ax.twinx()

    if "mu_nr" in df_main.columns:
        df_s2 = df_main[df_main["stage"] == 2].dropna(subset=["mu_nr"])
        if not df_s2.empty:
            sess = df_s2.groupby("session")["mu_nr"].median()
            cfg2 = STAGES.get(2)
            ax.plot(sess.index, sess.values, "o-",
                    color=cfg2.color if cfg2 else "salmon",
                    label="S2 mu_nr", ms=CFG["subj_ms"], lw=CFG["subj_lw"])
        df_s4 = df_main[df_main["stage"] == 4].dropna(subset=["mu_nr"])
        if not df_s4.empty:
            sess = df_s4.groupby("session")["mu_nr"].median()
            cfg4 = STAGES.get(4)
            ax.plot(sess.index, sess.values, "^-",
                    color=cfg4.color if cfg4 else "tomato",
                    label="S4 mu_nr", ms=CFG["subj_ms"], lw=CFG["subj_lw"])

    if "led_ms" in df_main.columns:
        df_s3 = df_main[df_main["stage"] == 3].dropna(subset=["led_ms"])
        if not df_s3.empty:
            sess = df_s3.groupby("session")["led_ms"].median()
            cfg = STAGES.get(3)
            ax2.plot(sess.index, sess.values, "s--",
                     color=cfg.color if cfg else "lightcoral",
                     label="S3 led_ms", ms=CFG["subj_ms"], lw=CFG["subj_lw"])

    ax.set_ylabel("mu_nr", color="steelblue")
    ax2.set_ylabel("led_ms (ms)", color=CFG["led_color"])
    ax.set_xlabel("Session")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=CFG["fs_label"],
              loc="upper left")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 300

    stage_arr = np.zeros(n, dtype=int)
    stage_arr[50:130] = 1
    stage_arr[130:200] = 2
    stage_arr[200:240] = 3
    stage_arr[240:270] = 4
    stage_arr[270:] = 5

    phase_arr = []
    for i in range(n):
        s = int(stage_arr[i])
        stage_start = int(np.argmax(stage_arr == s))
        phase_arr.append("warmup" if s in (2, 3, 4) and (i - stage_start) < 30
                         else "main")

    mu_nr = np.where(stage_arr == 2, np.linspace(0.0, 1.6, n),
                     np.where(stage_arr == 3, 1.6,
                     np.where(stage_arr == 4, np.linspace(1.6, 2.3, n),
                              np.where(stage_arr == 5, 2.3, 0.0))))
    mu_r = np.where(stage_arr >= 1, 7.7, 0.0)
    led_ms = np.where(stage_arr == 3,
                      np.linspace(5000, 800, n).astype(int),
                      np.where(stage_arr >= 4, 200, 5000))
    checkpoint_arr = np.where(stage_arr >= 2, stage_arr - 1, 0)
    floor_arr = np.where(stage_arr == 2, 0.5,
                         np.where(stage_arr == 3, 5000.0,
                                  np.where(stage_arr == 4, 2.0, 0.0)))
    rescue_arr = np.zeros(n, dtype=int)
    rescue_arr[270:280] = 1
    rescue_arr[288:298] = 1
    session_arr = np.searchsorted([60, 120, 180, 240], np.arange(n)) + 1

    correct_arr = rng.integers(0, 2, n)
    step_delta_arr = np.zeros(n)
    step_boost_arr = np.ones(n)
    _M, _tau, _nb = 4.0, 10.0, 30
    for _s in [2, 3, 4]:
        _mask = (stage_arr == _s) & (np.array(phase_arr) == "main")
        if _mask.any():
            _t = np.arange(1, _mask.sum() + 1)
            step_boost_arr[_mask] = np.where(
                _t <= _nb, _M * np.exp(-_t / _tau) + 1.0, 1.0)
            _base = 30.0 if _s == 3 else 0.005
            step_delta_arr[_mask] = np.abs(
                rng.normal(_base, _base * 0.3, _mask.sum()))

    df = pd.DataFrame({"trial":            np.arange(n),
                       "session":          session_arr,
                       "stage":            stage_arr,
                       "phase":            phase_arr,
                       "trial_correct":    correct_arr,
                       "mu_r":             mu_r,
                       "mu_nr":            mu_nr,
                       "led_ms":           led_ms,
                       "streak":           rng.integers(-5, 6, n),
                       "step_delta":       step_delta_arr,
                       "step_boost":       step_boost_arr,
                       "checkpoint":       checkpoint_arr,
                       "checkpoint_floor": floor_arr,
                       "delta_towers":     rng.integers(-4, 5, n),
                       "rescue":           rescue_arr})

    fig, axs = plt.subplots(8, 1, figsize=(13, 30))

    shade_phases(axs[0], df)
    plot_staircase(df, axs[0])

    shade_stages(axs[1], df)
    mark_checkpoints(axs[1], df)
    shade_phases(axs[1], df)
    shade_rescue(axs[1], df)
    plot_rolling_accuracy(df, axs[1], window=40, rescue_threshold=0.55)

    plot_streak(df, axs[2])

    shade_stages(axs[3], df)
    shade_phases(axs[3], df)
    plot_step(df, axs[3])

    plot_psychometric(df, axs[4])

    shade_stages(axs[5], df)
    mark_checkpoints(axs[5], df)
    shade_phases(axs[5], df)
    plot_staircase(df, axs[5])

    plot_stage_progression(df, axs[6])

    plot_difficulty_progression(df, axs[7])

    plt.tight_layout()
    plt.show()
