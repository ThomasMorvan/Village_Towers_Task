import numpy as np
import pandas as pd
import matplotlib.dates as mdates
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
    "reward_color":      "darkgoldenrod",
    "bias_color":        "purple",

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


# bias target is the same across stages
BIAS_TARGET = 0.10


def animal_bias(df, window):
    """Rolling side bias of the animal: |acc_right - acc_left| over the last
    `window` trials, matching WarmupTracker.bias in the difficulty
    controller. NaN until at least one trial of each side is in the window."""
    side_r = df["trial_side"] == "R"
    corr = df["trial_correct"].astype(float)

    def _acc(mask):
        c = (corr * mask).rolling(window, min_periods=1).sum()
        n = mask.astype(float).rolling(window, min_periods=1).sum()
        return c / n.replace(0, np.nan)

    return (_acc(side_r) - _acc(~side_r)).abs()


def to_time_axis(df):
    """Replace the trial index with minutes since session start so trial-axis
    plots are spaced by real time. Returns (df, xlabel). Falls back to the
    trial index when no usable TRIAL_START timestamps are present."""
    df = df.reset_index(drop=True)
    if "TRIAL_START" in df.columns and df["TRIAL_START"].notna().any():
        ts = df["TRIAL_START"].ffill().bfill()
        return (df.assign(trial=(ts - ts.iloc[0]) / 60.0),
                "Time in session (min)")
    return df, "Trial"


def _dx(x):
    """Typical spacing between adjacent x positions (1.0 for a trial index;
    median gap when x is time, so shading/bars stay sized to real spacing)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 1.0
    d = np.diff(np.sort(np.unique(x)))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def shade_phases(ax, df):
    """warmup phase shade."""
    if "phase" not in df.columns:
        return
    df = df.reset_index(drop=True)
    trials = df["trial"].tolist()
    phases = df["phase"].tolist()
    h = _dx(trials) / 2
    start, cur = trials[0], phases[0]
    for t, p in zip(trials[1:], phases[1:]):
        if p != cur:
            if cur == "warmup":
                ax.axvspan(start - h, t - h,
                           alpha=CFG["warmup_alpha"],
                           color=CFG["warmup_color"], zorder=0,
                           lw=0)
            start, cur = t, p
    if cur == "warmup":
        ax.axvspan(start - h, trials[-1] + h,
                   alpha=CFG["warmup_alpha"],
                   color=CFG["warmup_color"], zorder=0,
                   lw=0)


def shade_rescue(ax, df):
    """Shade rescue blocks in red."""
    if "rescue" not in df.columns:
        return
    trials = df["trial"].tolist()
    rescues = df["rescue"].fillna(0).astype(int).tolist()
    h = _dx(trials) / 2
    in_block = False
    for t, r in zip(trials, rescues):
        if r and not in_block:
            start = t
            in_block = True
        elif not r and in_block:
            ax.axvspan(start - h, t - h,
                       color=CFG["rescue_color"], alpha=CFG["rescue_alpha"],
                       zorder=1, lw=0, label="Rescue")
            in_block = False
    if in_block:
        ax.axvspan(start - h, trials[-1] + h,
                   color=CFG["rescue_color"], alpha=CFG["rescue_alpha"],
                   zorder=1, lw=0, label="Rescue")


def _stage_label(stage, df_seg):
    """S0 splits into two steps (ROI proximity / poke); label accordingly."""
    label = f"S{int(stage)}"
    if int(stage) == 0 and "proximity_trigger" in df_seg.columns:
        vals = df_seg["proximity_trigger"].dropna()
        if not vals.empty:
            label += "·ROI" if bool(vals.iloc[-1]) else "·poke"
    return label


def shade_stages(ax, df):
    """Background color for each stage."""
    df = df.reset_index(drop=True)
    h = _dx(df["trial"]) / 2
    cur_s = df["stage"].iloc[0]
    start_i = 0
    start_t = df["trial"].iloc[0]
    for i, row in df.iterrows():
        if row["stage"] != cur_s:
            cfg = STAGES.get(int(cur_s))
            ax.axvspan(start_t - h, row["trial"] - h,
                       alpha=CFG["stage_alpha"],
                       color=cfg.color if cfg else "w", zorder=0, lw=0)
            ax.text((start_t + row["trial"]) / 2, 1.02,
                    _stage_label(cur_s, df.iloc[start_i:i]),
                    ha="center", fontsize=CFG["fs"],
                    transform=ax.get_xaxis_transform())
            start_i, start_t, cur_s = i, row["trial"], row["stage"]
    cfg = STAGES.get(int(cur_s))
    ax.axvspan(start_t - h, df["trial"].iloc[-1] + h,
               alpha=CFG["stage_alpha"],
               color=cfg.color if cfg else "w", zorder=0, lw=0)
    ax.text((start_t + df["trial"].iloc[-1]) / 2, 1.02,
            _stage_label(cur_s, df.iloc[start_i:]),
            ha="center", fontsize=CFG["fs"],
            transform=ax.get_xaxis_transform())


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
        df_exp = df.dropna(subset=["mu_r", "mu_nr"])
        if not df_exp.empty:
            ax.plot(df_exp["trial"], df_exp["mu_r"] - df_exp["mu_nr"],
                    color=CFG["delta_color"], lw=CFG["mu_nr_lw"],
                    zorder=2, label=r"$\Delta\mu$")

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


def plot_rolling_accuracy(df, ax, window: int = 100):
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

    if "trial_side" in df.columns:
        bias = animal_bias(df, window)
        ax.plot(df["trial"], bias, color=CFG["bias_color"], lw=CFG["acc_lw"],
                alpha=0.8, label="Animal bias |accR-accL|", zorder=2)
        ax.axhline(BIAS_TARGET, color=CFG["bias_color"], ls=":",
                   lw=CFG["thr_lw"], alpha=CFG["thr_alpha"],
                   label=f"Target bias ({BIAS_TARGET})")

    if "stage" in df.columns and len(df):
        stages = df["stage"].astype(int).to_numpy()

        def _per_trial(attr, skip_zero=False):
            out = []
            for s in stages:
                cfg = STAGES.get(s)
                v = getattr(cfg, attr) if cfg is not None else None
                if v is None or (skip_zero and v <= 0):
                    out.append(np.nan)
                else:
                    out.append(v)
            return np.array(out, dtype=float)

        tgt = _per_trial("advance_threshold", skip_zero=True)
        rsc = _per_trial("rescue_threshold")
        if not np.isnan(tgt).all():
            ax.plot(df["trial"], tgt, drawstyle="steps-post",
                    color=CFG["thr_s23_color"], ls=CFG["target_ls"],
                    lw=CFG["thr_lw"], alpha=CFG["thr_alpha"],
                    label="Target accuracy", zorder=2)
        if not np.isnan(rsc).all():
            ax.plot(df["trial"], rsc, drawstyle="steps-post",
                    color=CFG["rescue_color"], ls=CFG["rescue_thr_ls"],
                    lw=CFG["rescue_thr_lw"], alpha=CFG["rescue_thr_alpha"],
                    label="Rescue threshold", zorder=2)

    if "reward_mult" in df.columns and (df["reward_mult"] > 1.0).any():
        b = df[df["reward_mult"] > 1.0]
        jp = (b["jackpot"] == 1) if "jackpot" in b.columns else pd.Series(
            False, index=b.index)
        eff = b[~jp]
        if not eff.empty:
            ax.scatter(eff["trial"], [1.07] * len(eff), marker=".", s=18,
                       color=CFG["reward_color"], alpha=0.8, zorder=3,
                       label="Reward boost")
        if jp.any():
            ax.scatter(b.loc[jp, "trial"], [1.07] * int(jp.sum()), marker="*",
                       s=45, color=CFG["reward_color"], edgecolors="k",
                       linewidths=0.3, zorder=4, label="Jackpot")

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


def _step_bars(ax, sub_df, boost_series, ok_color, err_color, width=1.0):
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
           width=width, label="correct")
    ax.bar(sub_df.loc[~ok, "trial"], base[~ok],
           color=err_color, alpha=CFG["streak_alpha"],
           width=width, label="incorrect")
    boosted = extra > 1e-9
    if boosted.any():
        ax.bar(sub_df.loc[boosted, "trial"], extra[boosted],
               bottom=base[boosted], color="gold", alpha=0.7, width=width,
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

    w = _dx(df["trial"])
    boost = df["step_boost"] if "step_boost" in df.columns else None
    df_dens = df[df["stage"].isin([2, 4])] if "stage" in df.columns else df
    _step_bars(ax, df_dens,
               boost if boost is not None else df_dens["step_delta"],
               CFG["streak_ok"], CFG["streak_err"], width=w)

    df_ms = df[df["stage"] == 3] if "stage" in df.columns else pd.DataFrame()
    _step_bars(ax2, df_ms,
               (boost if boost is not None
                else df_ms["step_delta"] if not df_ms.empty
                else pd.Series(dtype=float)),
               CFG["led_color"], "darkorange", width=w)

    df_cue = df[df["stage"] == 1] if "stage" in df.columns else pd.DataFrame()
    if not df_cue.empty:
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.10))
        _step_bars(ax3, df_cue,
                   boost if boost is not None else df_cue["step_delta"],
                   STAGES[1].color, "darkgreen", width=w)
        ax3.set_ylabel("Δ cue intensity (PWM)", color=STAGES[1].color)
        ax3.tick_params(axis="y", labelcolor=STAGES[1].color)

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


def plot_lr_bars(df, ax):
    """Correct / incorrect trial counts per rewarded side (L vs R)."""
    if "trial_side" not in df.columns:
        ax.text(0.5, 0.5, "No trial_side data", ha="center", va="center",
                transform=ax.transAxes)
        return
    d = df.dropna(subset=["trial_side", "trial_correct"])
    if d.empty:
        ax.text(0.5, 0.5, "No trial data", ha="center", va="center",
                transform=ax.transAxes)
        return
    ok = d["trial_correct"].astype(bool)
    sides = ["L", "R"]
    correct = [int(((d["trial_side"] == s) & ok).sum()) for s in sides]
    incorrect = [int(((d["trial_side"] == s) & ~ok).sum()) for s in sides]
    x = np.arange(len(sides))
    w = 0.38
    bars = [(x - w / 2, correct, CFG["streak_ok"], "correct"),
            (x + w / 2, incorrect, CFG["streak_err"], "incorrect")]
    for xs, vals, color, label in bars:
        ax.bar(xs, vals, w, color=color, alpha=0.7, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v, str(v), ha="center", va="bottom",
                    fontsize=CFG["fs"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Left", "Right"])
    ax.set_ylabel("Trials")
    ax.legend(fontsize=CFG["fs"], loc="upper right")


def _advance_criteria(df, window, settings=None):
    """Diagnostic advance criteria."""
    cur = int(df["stage"].iloc[-1])
    cfg = STAGES.get(cur)
    phase = df["phase"].iloc[-1] if "phase" in df.columns else "main"
    step = cfg.staircase.grad_tol(settings) if cfg is not None else 0.0

    # contiguous block of the current stage
    chg = (df["stage"] != df["stage"].shift()).cumsum()
    seg = df[chg == chg.iloc[-1]]
    corr = df["trial_correct"].astype(float)
    rolling_acc = float(corr.tail(window).mean()) if len(corr) else 0.0

    def last(col, default=np.nan):
        return float(df[col].iloc[-1]) if col in df.columns else default

    def acc_of(d):
        return float(d["trial_correct"].astype(float).mean()) if len(d) else 0.0

    def bias_of(d):
        return (float(animal_bias(d, max(len(d), 1)).iloc[-1])
                if len(d) else 1.0)

    rows = []
    if phase == "warmup" and cfg is not None and cfg.has_warmup:
        seg_w = seg[seg["phase"] == "warmup"]
        wn = int(last("warmup_trial", len(seg_w)))
        rows = [
            ("Warmup trials", wn, cfg.warmup_min_trials or 0, 0, False,
             "int", 0.0),
            ("Warmup acc", acc_of(seg_w), cfg.warmup_acc_threshold or 0.0,
             0.0, False, "pct", 0.0),
            ("Warmup bias", bias_of(seg_w),
             cfg.warmup_bias_threshold or BIAS_TARGET, 0.5, True, "pct", 0.0),
        ]
        return f"S{cur} warmup → enter main", rows

    if cur == 0:
        rows = [("Trials", len(df), 40, 0, False, "int", 0.0)]
    elif cur == 1:
        empr = last("empR")
        bias = abs(empr - 0.5) if not np.isnan(empr) and empr >= 0 \
            else bias_of(seg)
        rows = [
            ("Accuracy", rolling_acc, cfg.advance_threshold, 0.0, False,
             "pct", 0.0),
            ("Bias", bias, BIAS_TARGET, 0.5, True, "pct", 0.0),
            ("Cue intensity", last("light_intensity"), cfg.staircase.target,
             cfg.staircase.start, True, "int", step),
        ]
    elif cur in (2, 4):
        rows = [
            ("Accuracy", rolling_acc, cfg.advance_threshold, 0.0, False,
             "pct", 0.0),
            ("mu_nr", last("mu_nr"), cfg.staircase.target,
             cfg.staircase.start, False, "f4", step),
        ]
    elif cur == 3:
        rows = [
            ("Accuracy", rolling_acc, cfg.advance_threshold, 0.0, False,
             "pct", 0.0),
            ("LED ms", last("led_ms"), cfg.staircase.target,
             cfg.staircase.start, True, "int", step),
        ]
    name = cfg.name if cfg else "?"
    return f"S{cur} {name} → advance", rows


def plot_stage_diagnostic(df, ax, window=40, settings=None):
    """Offline view of the HUD advance criteria for the current stage:
    one bar per criterion (progress start→target), green if met else red."""
    if df.empty or "stage" not in df.columns:
        ax.text(0.5, 0.5, "No stage data", ha="center", va="center",
                transform=ax.transAxes)
        return
    title, rows = _advance_criteria(df, window, settings)
    ax.set_title(title, fontsize=CFG["fs_label"])
    if not rows:
        ax.text(0.5, 0.5, "Final stage\nnothing to advance", ha="center",
                va="center", transform=ax.transAxes, fontsize=CFG["fs_label"])
        ax.axis("off")
        return

    def fmt(v, kind):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "?"
        return {"pct": f"{v * 100:.0f}%", "int": f"{v:.0f}",
                "f4": f"{v:.4f}"}.get(kind, str(v))

    names = []
    for i, (name, val, target, start, lower, kind, tol) in enumerate(rows):
        nan = isinstance(val, float) and np.isnan(val)
        met = (not nan) and ((val <= target + tol) if lower
                             else (val >= target - tol))
        if nan or start == target:
            prog = 1.0 if met else 0.0
        else:
            prog = (val - start) / (target - start)
        prog = float(np.clip(prog, 0.0, 1.0))
        color = CFG["streak_ok"] if met else CFG["streak_err"]
        ax.barh(i, prog, color=color, alpha=0.7, height=0.6)
        mark = "✓" if met else "✗"
        ax.text(min(prog, 1.0) + 0.02, i,
                f"{fmt(val, kind)} / {fmt(target, kind)} {mark}",
                va="center", ha="left", fontsize=CFG["fs"], color=color)
        names.append(name)
    ax.axvline(1.0, color="gray", ls="--", lw=1.0)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=CFG["fs_label"])
    ax.set_xticks([0, 1.0])
    ax.set_xticklabels(["start", "target"], fontsize=CFG["fs"])


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

    def _trial_x(g):
        """x = session number + within-session offset in [-0.4, 0.4], so each
        session's trace is centered on its integer tick (and its median dot)."""
        g = g.sort_values(["session", "trial"])
        n = g.groupby("session")["mu_nr"].transform("size")
        frac = g.groupby("session").cumcount() / n.where(n > 1, 1)
        return g["session"] + (frac - 0.5) * 0.8, g

    if "mu_nr" in df_main.columns:
        df_s2 = df_main[df_main["stage"] == 2].dropna(subset=["mu_nr"])
        if not df_s2.empty:
            cfg2 = STAGES.get(2)
            color = cfg2.color if cfg2 else "salmon"
            x, g = _trial_x(df_s2)
            ax.plot(x, g["mu_nr"], "-", color=color, alpha=0.25, lw=0.8)
            sess = df_s2.groupby("session")["mu_nr"].median()
            ax.plot(sess.index, sess.values, "o-", color=color,
                    label="S2 mu_nr", ms=CFG["subj_ms"], lw=CFG["subj_lw"])
        df_s4 = df_main[df_main["stage"] == 4].dropna(subset=["mu_nr"])
        if not df_s4.empty:
            cfg4 = STAGES.get(4)
            color = cfg4.color if cfg4 else "tomato"
            x, g = _trial_x(df_s4)
            ax.plot(x, g["mu_nr"], "-", color=color, alpha=0.25, lw=0.8)
            sess = df_s4.groupby("session")["mu_nr"].median()
            ax.plot(sess.index, sess.values, "^-", color=color,
                    label="S4 mu_nr", ms=CFG["subj_ms"], lw=CFG["subj_lw"])

    if "led_ms" in df_main.columns:
        df_s3 = df_main[df_main["stage"] == 3].dropna(subset=["led_ms"])
        if not df_s3.empty:
            sess = df_s3.groupby("session")["led_ms"].median()
            cfg = STAGES.get(3)
            ax2.plot(sess.index, sess.values, "s--",
                     color=cfg.color if cfg else "lightcoral",
                     label="S3 led_ms", ms=CFG["subj_ms"], lw=CFG["subj_lw"])

    sessions = sorted(df_main["session"].dropna().unique())
    for s in sessions:
        ax.axvline(s - 0.5, color="0.85", lw=0.6, zorder=0)
    if sessions:
        ax.axvline(sessions[-1] + 0.5, color="0.85", lw=0.6, zorder=0)

    ax.set_ylabel("mu_nr", color="steelblue")
    ax2.set_ylabel("led_ms (ms)", color=CFG["led_color"])
    ax.set_xlabel("Session")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=CFG["fs_label"],
              loc="upper left")


def demo_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic session df covering stages 0-5."""
    rng = np.random.default_rng(seed)

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

    # Trial timestamps: 2-8 s apart, with a couple of longer pauses, so the
    # time-spaced x-axis differs visibly from the trial index.
    gaps = rng.uniform(2.0, 8.0, n)
    gaps[100] += 120.0
    gaps[220] += 240.0
    trial_start = 1.7e9 + np.cumsum(gaps)

    correct_arr = rng.integers(0, 2, n)
    step_delta_arr = np.zeros(n)
    step_boost_arr = np.ones(n)
    _M, _tau, _nb = 4.0, 10.0, 30
    _base_by_stage = {1: 15.0, 2: 0.005, 3: 30.0, 4: 0.005}
    for _s in [1, 2, 3, 4]:
        _mask = (stage_arr == _s) & (np.array(phase_arr) == "main")
        if _mask.any():
            _t = np.arange(1, _mask.sum() + 1)
            step_boost_arr[_mask] = np.where(
                _t <= _nb, _M * np.exp(-_t / _tau) + 1.0, 1.0)
            _base = _base_by_stage[_s]
            step_delta_arr[_mask] = np.abs(
                rng.normal(_base, _base * 0.3, _mask.sum()))

    light_intensity = np.where(stage_arr < 1, 255,
                               np.where(stage_arr == 1,
                                        np.linspace(255, 30, n).astype(int), 0))
    emp_r = np.clip(0.5 + rng.normal(0, 0.07, n), 0.05, 0.95)

    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(session_arr - 1,
                                                           unit="D")
    return pd.DataFrame({"trial":            np.arange(n),
                         "date":             dates,
                         "session":          session_arr,
                         "TRIAL_START":      trial_start,
                         "stage":            stage_arr,
                         "phase":            phase_arr,
                         "trial_correct":    correct_arr,
                         "mu_r":             mu_r,
                         "mu_nr":            mu_nr,
                         "led_ms":           led_ms,
                         "light_intensity":  light_intensity,
                         "empR":             emp_r,
                         "streak":           rng.integers(-5, 6, n),
                         "step_delta":       step_delta_arr,
                         "step_boost":       step_boost_arr,
                         "checkpoint":       checkpoint_arr,
                         "checkpoint_floor": floor_arr,
                         "delta_towers":     rng.integers(-4, 5, n),
                         "trial_side":       np.where(rng.random(n) < 0.5,
                                                      "L", "R"),
                         "rescue":           rescue_arr})


def online_figure(df):
    df_t, xlabel = to_time_axis(df)
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 8),
                                             layout="constrained")
    shade_stages(a1, df_t)
    mark_checkpoints(a1, df_t)
    shade_phases(a1, df_t)
    plot_staircase(df_t, a1)
    shade_stages(a2, df_t)
    mark_checkpoints(a2, df_t)
    shade_phases(a2, df_t)
    shade_rescue(a2, df_t)
    plot_rolling_accuracy(df_t, a2)
    plot_streak(df, a3)
    shade_stages(a4, df)
    shade_phases(a4, df)
    plot_step(df, a4)
    a1.set_xlabel(xlabel)
    a2.set_xlabel(xlabel)
    a3.set_xlabel("Trial")
    a4.set_xlabel("Trial")
    fig.suptitle("Online plot")
    return fig


def session_figure(df):
    df, xlabel = to_time_axis(df)
    fig, axd = plt.subplot_mosaic(
        [["stair", "stair"],
         ["acc", "acc"],
         ["psy", "psy"],
         ["lr", "diag"]],
        figsize=(11, 13),
        gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5]})
    a1, a2 = axd["stair"], axd["acc"]
    shade_stages(a1, df)
    mark_checkpoints(a1, df)
    shade_phases(a1, df)
    plot_staircase(df, a1)
    shade_stages(a2, df)
    mark_checkpoints(a2, df)
    shade_phases(a2, df)
    shade_rescue(a2, df)
    plot_rolling_accuracy(df, a2, window=40)
    plot_psychometric(df, axd["psy"])
    plot_lr_bars(df, axd["lr"])
    plot_stage_diagnostic(df, axd["diag"], window=40)
    a1.set_xlabel(xlabel)
    a2.set_xlabel(xlabel)
    fig.suptitle("Session plot")
    fig.tight_layout()
    return fig


def plot_trials_per_day(df, ax):
    """Trials per session date on a real date axis (auto-thinned ticks).
    Mirrors the bar panel in SubjectPlot."""
    counts = df["date"].value_counts(sort=False)
    counts.index = pd.to_datetime(counts.index)
    ax.bar(counts.index, counts.values, width=0.8)
    loc = mdates.AutoDateLocator(maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_ylabel("Number of trials")
    ax.set_xlabel("Date")


def subject_figure(df):
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11, 11))
    plot_trials_per_day(df, a0)
    plot_stage_progression(df, a1)
    plot_difficulty_progression(df, a2)
    fig.suptitle("Subject plot")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = demo_df()
    online_figure(df)
    session_figure(df)
    subject_figure(df)
    plt.show()
