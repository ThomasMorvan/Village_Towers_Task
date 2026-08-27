import glob
import os
import time
import traceback

import numpy as np
import pandas as pd

try:  # for debugging
    from village.custom_classes.after_session_base import AfterSessionBase
    from village.manager import manager
    from village.scripts.log import log
except ImportError:
    AfterSessionBase, manager, log = object, None, None


class QCResult:
    def __init__(self):
        self.raw_exists = False
        self.session_exists = False
        self.tracking_exists = False
        self.video_exists = False

        # default to "ok, nothing to say"
        self.pokes_ok: tuple[bool, str | None] = (True, None)
        self.tracking_ok: tuple[bool, str | None] = (True, None)
        self.video_ok: tuple[bool, str | None] = (True, None)
        self.behavior_ok: tuple[bool, str | None] = (True, None)

    @property
    def flags(self) -> list[str]:
        checks = (self.pokes_ok, self.tracking_ok,
                  self.video_ok, self.behavior_ok)
        return [msg for ok, msg in checks if not ok and msg]

    @property
    def ok(self) -> bool:
        return not self.flags

    def __str__(self) -> str:
        return "\n".join(self.flags) if self.flags else "ok"


class SessionQC:
    """Check the health of a TowersTask session from its raw csv and video."""
    TASK = "TowersTask"
    PORTS = {1: "LEFT", 2: "MIDDLE", 3: "RIGHT"}
    CHOICE = (1, 3)

    # thresholds
    GAP_S = 0.1  # an inter-frame gap above this counts as a stall
    STALL_MAX = 0.1  # >10% lost to stalls -> degraded
    FPS_MIN = 15.0  # effective fps below this -> degraded (nominal ~30)
    ACC_MIN = 0.55  # evidence-trial accuracy <= this -> at chance
    MIN_TRIALS = 20  # need this many trials to trust the accuracy
    MIN_VIDEO_BYTES = 100_000  # smaller/zero .mp4 -> encoder died -> no video

    def __init__(self, raw_path: str,
                 data_dir: str | None = None,
                 no_video: bool = False):
        self.raw_path = raw_path
        self.data_dir = data_dir or os.path.dirname(os.path.dirname(
                                                    os.path.dirname(raw_path)))
        self.no_video = no_video

        self.result = QCResult()

    def run(self) -> "QCResult":
        if self.TASK not in self.raw_path:
            return self.result  # not a TowersTask session, nothing to check

        # check files exist.
        dirname = os.path.dirname(self.raw_path)
        base = os.path.basename(self.raw_path)[: -len("_RAW.csv")]
        subject = base.split("_")[0]
        videos = os.path.join(self.data_dir, "videos", subject)

        session_path = os.path.join(dirname, base + ".csv")
        tracking_path = os.path.join(videos, base + ".csv")
        video_path = os.path.join(videos, base + ".mp4")

        self.result.raw_exists = os.path.isfile(self.raw_path)
        self.result.session_exists = os.path.isfile(session_path)
        self.result.tracking_exists = os.path.isfile(tracking_path)
        self.result.video_exists = os.path.isfile(video_path)

        if self.result.raw_exists:
            # in raw_session, we can check the pokes.
            self.result.pokes_ok = self.check_poke(self.raw_path)

        if self.result.session_exists:
            # in session, we can check the accuracy.
            self.result.behavior_ok = self.check_behavior(session_path)

        if self.result.tracking_exists:
            # in tracking, we can check the tracking health.
            self.result.tracking_ok = self.check_tracking(tracking_path)

        if self.result.video_exists:
            # in video, we can check the video file size.
            self.result.video_ok = self.check_video(video_path)

        return self.result

    def check_video(self, path: str) -> tuple[bool, str | None]:
        if self.no_video:
            return True, None
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        fp = os.path.basename(path)
        if size == 0:
            return False, f"[QC VIDEO]: {fp} was not written"
        if size < self.MIN_VIDEO_BYTES:
            return False, f"[QC VIDEO]: {fp} is {size} bytes"
        return True, None

    def check_tracking(self, path: str) -> tuple[bool, str | None]:
        """Check the tracking health from the tracking csv file."""
        if self.no_video:
            return True, None
        try:
            tracking = pd.read_csv(path, sep=";")
        except Exception:
            return False, f"[QC TRACKING]: {path} could not be read"

        if "timestamp" not in tracking.columns:
            return False, f"[QC TRACKING]: {path} has no timestamp column"

        # check stability
        timestamps = tracking["timestamp"].to_numpy(dtype=float)
        t = np.sort(np.asarray(timestamps, dtype=float))
        t = t[np.isfinite(t)]
        dt = np.diff(t)
        dur = t[-1] - t[0]

        # effective fps
        eff_fps = len(t) / dur if dur > 0 else np.nan
        fps_degraded = eff_fps < self.FPS_MIN

        # stall fraction
        freezes = dt[dt > self.GAP_S]
        stall_frac = freezes.sum() / dur if dur > 0 else np.nan
        stalling = stall_frac > self.STALL_MAX

        if fps_degraded or stalling:
            return False, (f"[QC TRACKING]: Low effective FPS ({eff_fps:.1f}),"
                           f" {stall_frac:.0%} session stalled, "
                           f"{len(freezes)} freezes (worst {dt.max():.1f} s)")
        return True, None

    def check_behavior(self, path: str) -> tuple[bool, str | None]:
        try:
            data = pd.read_csv(path, sep=";")
        except Exception:
            return False, f"[QC BEHAVIOR]: {path} could not be read"

        # check that the columns we need are present
        if not {"trial_correct", "L LEDs", "R LEDs"}.issubset(data.columns):
            return False, f"[QC BEHAVIOR]: {path} has wrong or no columns"

        def _n_leds(cell) -> int:
            """Count of a 'L LEDs'/'R LEDs' cell."""
            s = str(cell).strip("[] ")
            return s.count(",") + 1 if s and s[0].isdigit() else 0

        # an evidence trial is one where the LED-strip towers actually lit;
        # a stalled camera can't fire them, so accuracy collapses toward
        # chance on those trials specifically
        ev = (data["L LEDs"].map(_n_leds) + data["R LEDs"].map(_n_leds)) > 0
        n_trials = int(ev.sum())
        if n_trials < self.MIN_TRIALS:
            return True, None  # too few evidence trials to trust the accuracy

        acc = float(data.loc[ev, "trial_correct"].mean())
        if acc <= self.ACC_MIN:
            return False, (f"[QC BEHAVIOR]: Low accuracy ({acc:.0%}) on "
                           f"{n_trials} evidence trials")

        # compute bias, etc

        return True, None

    def check_poke(self, path: str) -> tuple[bool, str | None]:
        """Per-port IN counts and how many choice trials got no side poke from
        RAW Bpod event log."""
        try:
            raw = pd.read_csv(path, sep=";")
        except Exception:
            return False, f"[QC POKE]: {path} could not be read"

        if "MSG" not in raw.columns:
            return False, f"[QC POKE]: {path} has no MSG column"

        is_ok = True
        issues: list[str] = []

        n_trials = int(raw.loc[raw["MSG"] == "TRIAL_START", "TRIAL"].nunique())
        counts = {p: int((raw["MSG"] == f"Port{p}In").sum())
                  for p in self.PORTS}
        trials_with = {p: raw.loc[raw["MSG"] == f"Port{p}In", "TRIAL"]
                       .nunique() for p in self.PORTS}
        side_trials = set(raw.loc[raw["MSG"].isin(["Port1In", "Port3In"]),
                                  "TRIAL"])
        all_trials = set(raw.loc[raw["MSG"] == "TRIAL_START", "TRIAL"])
        no_choice = len(all_trials - side_trials)

        if n_trials == 0:
            return False, "[QC POKE]: No trials found"

        if counts[2] == 0:
            return False, f"[QC POKE]: Port2 (0 pokes in {n_trials} trials)"

        other = {1: 3, 3: 1}
        for p in self.CHOICE:
            tw, to = trials_with[p], trials_with[other[p]]
            if counts[p] == 0 and counts[other[p]] > 0:
                issues.append(f"[QC POKE]: Port{p} DEAD: 0 pokes vs "
                              f"{counts[other[p]]} on {self.PORTS[other[p]]}")
                is_ok = False
            frac = (tw / n_trials, to / n_trials)
            if (n_trials >= self.MIN_TRIALS
                    and frac[0] < 0.10
                    and frac[1] > 0.30):
                _str = (f"[QC POKE]: Port{p}: {tw} pokes in {n_trials} trials "
                        f"(vs {to} in {n_trials} trials in Port{other[p]})")
                issues.append(_str)
                is_ok = False

        if no_choice / n_trials > 0.30:
            _str = f"[QC POKE]: {no_choice}/{n_trials} TRIALS NO SIDE POKES"
            issues.append(_str)
            is_ok = False

        return is_ok, "\n".join(issues) if issues else None


class AfterSession(AfterSessionBase):
    """QC the session that just ended, then sync (super)."""

    def run(self) -> None:
        try:
            self.quality_check()
        except Exception as e:
            log.error("after-session quality check failed: %s", e,
                      exception=traceback.format_exc())
        super().run()

    def quality_check(self) -> None:
        task = manager.task
        result = SessionQC(task.raw_session_path, self.data_directory).run()
        name = os.path.basename(task.session_path) or task.name
        if result.ok:
            log.info(f"QC {name}: ok", subject=task.subject)
        else:
            log.alarm(f"QC {name}: {result}",
                      subject=task.subject, repeat=True)


if __name__ == "__main__":
    animal = 'XO'
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data")
    path = os.path.join(data_dir, "sessions", animal, "**", "*_RAW.csv")
    sessions = glob.glob(path, recursive=True)
    # sessions = sessions[-2:]
    for session in sessions:
        result = SessionQC(session, no_video=True).run()
        if not result.ok:
            print("  ", os.path.basename(session))
            print(result)
            print()

    start = time.perf_counter()
    for session in sessions:
        SessionQC(session, no_video=True).run()
    elapsed = time.perf_counter() - start
    print(f"{len(sessions)} sessions in {elapsed:.2f}s "
          f"({1000 * elapsed / len(sessions):.1f} ms/session)")
