r"""Background subtraction for CORRIDOR areas.

Initial problem: simple thresholding of the raw frame is too sensitive to
lighting changes, bad illumination, reflections, dirt, etc.
    --> Use a background subtraction method that is robust to these issues.
        --> Pick a few frames (30 frames with no mice detected inside the
            corridor), median them and do the thresholding by subtracting this
            median background from the current frame.
            --> That does not work well if there are scene changes
                (lights on/off, dirt, etc.).
                --> Add a control ROI (outside of corridor) that can detect
                    scene changes. If a scene change is detected, delete
                    previous background and start learning background again.
                    --> This causes problems if a mouse is in the corridor when
                        the scene changes, because the background will start
                        learning the mouse as part of the scene.
                        --> At this point I gave up and changed method.

The new idea is based on this trick:
The mice are dark (~0), the corridor floor is light (~255).
1) For every pixel, we can *track the brightest value seen so far*; as the mice
are dark, they can't push this value up
    --> mouse can't be absorbed into the background.
2) To track slow real changes (dirt, etc.), we periodically (LEAK_EVERY)
"leak" to slowly decrease its value. Rule 1 pushes this value up as soon as it
becomes bright again, so the leak only accumulates where something
dark is present.
https://en.wikipedia.org/wiki/Envelope_detector#Diode_detector


Background computation logic (diagram not to scale).

         ^ increase because biggest value seen so far
P            _________
X      _____/^   ^    \   LEAK                          ___________
         ^   |   |     \_______                         ^          \
V        |   |   |             \   LEAK                 |           \
A        |   |   |              \______                 |            .....
L        |   |   |                     \                |  ^ animal leaves,
U        |   |   |                      \_______        |  goes back up
E        |   |   |    leak because mouse        \       |
         |   |   |    is present                 \______|

                                   TIME

3) We then threshold based on the difference between the current frame and the
computed background to detect mice.

4) A control ROI (a spot in camera view that can't contain a mouse) is used to
detect global scene changes (someone turns lights on/off).
The control ROI gives a reference for the scene's overall brightness and any
significant deviation from this reference indicates a global scene change,
which triggers a relevel of the background (scale up/down if brighter/darker).

Possible problem: if a mouse stays at the same spot for very long, it may
eventually be absorbed into the background due to the leak
mechanism (~3.5 h at LEAK_EVERY=1800), this is not likely to happen in
practice. To prevent that, we trigger an alarm after STUCK_MINUTES.

One background for day and one for night. The backgrounds are saved to disk
and restored on subsequent runs.
"""


import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from village.custom_classes.camera_detection_base import (
        CorridorDetectionBase,
        CustomDetectionParam,
    )
    from village.scripts.log import log
    from village.settings import Color, settings
except ImportError:
    CorridorDetectionBase = object

    class CustomDetectionParam:  # dummy fallback
        def __init__(self, *args, **kwargs) -> None:
            pass

    settings = None
    log = None
    Color = None


# Background substractor params
LEAK_EVERY = 1800
OPEN_KERNEL = 3

# pos of the control ROI: spot that can never have a mouse in it, so any change
# there is a scene change, not a mouse
CONTROL_ROI = (300, 200, 360, 260)  # (x1, y1, x2, y2)
CONTROL_TOLERANCE = 0.02  # relative brightness change that triggers a relevel
STUCK_MINUTES = 120  # Max duration area continuously occupied before alarm

# FIXME delete because cam.thresholds are tuned for the old method,
# once we trust the new method, cam.thresholds will be used with UI values.
DIFF_THRESHOLD = 50
DIFF_THRESHOLD_NIGHT = 50


class BackgroundSubtractor:
    """Background subtraction.
    Params:
        threshold: absdiff cutoff (0-255) above which a pixel is foreground.
        leak_every: frames per one grey level of decay. Lower --> adapts faster
        but absorbs a stationary animal sooner. 0 disables the leak.
        dark_subjects: True when animals are darker than the background
                       (DETECTION_COLOR=BLACK). Flips if WHITE.
        open_kernel: size of an optional cv2.MORPH_OPEN pass on the output
                     mask to strip single-pixel speckle noise. 0 = disabled."""

    def __init__(self, threshold: float = 25,
                 open_kernel: int = OPEN_KERNEL,
                 leak_every: int = LEAK_EVERY,
                 dark_subjects: bool = True) -> None:
        self.threshold = threshold
        self.open_kernel = open_kernel
        self.leak_every = leak_every
        self.dark_subjects = dark_subjects
        self._bgs: dict[bool, np.ndarray | None] = {False: None, True: None}
        self._ns: dict[bool, int] = {False: 0, True: 0}

    def apply(self, gray: np.ndarray, night: bool = False) -> np.ndarray:
        """Feed one grayscale frame, get back the foreground mask."""
        self._ns[night] += 1
        bg = self._bgs[night]

        if bg is None or bg.shape != gray.shape:
            # Nothing learned yet, or the area was resized in the GUI,
            # start fresh, so ensure no mouse is in the corridor.
            bg = self._bgs[night] = gray.copy()
        else:
            # Leak first, so a mouse-free pixel lands exactly on the frame
            # value rather than one level under it.
            if self.leak_every and self._ns[night] % self.leak_every == 0:
                cv2.add(bg, -1 if self.dark_subjects else 1, dst=bg)
            envelope = np.maximum if self.dark_subjects else np.minimum
            envelope(bg, gray, out=bg)

        # compute absdiff from the current background, threshold to get a mask
        diff = cv2.absdiff(gray, bg)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        if self.open_kernel:
            kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def relevel(self, gain: float, night: bool | None = None) -> None:
        """Rescales the stored background by global brightness."""
        for ctx in ([False, True] if night is None else [night]):
            if self._bgs[ctx] is not None:
                self._bgs[ctx] = self._scale(self._bgs[ctx], gain)

    def carry_over(self, gain: float, to_night: bool) -> bool:
        """Seeds one light context's background from the
        other, rescaled to it."""
        src = self._bgs[not to_night]
        if src is None or self._bgs[to_night] is not None:
            return False
        self._bgs[to_night] = self._scale(src, gain)
        return True

    def flush(self, night: bool | None = None) -> None:
        """Drops the background entirely, relearning it from the next frame.
        None (default) flushes both day and night. Should only be called by
        hand from UI: nothing automatic can tell a sleeping mouse from a
        corridor that really is empty."""
        for ctx in ([False, True] if night is None else [night]):
            self._ns[ctx] = 0
            self._bgs[ctx] = None

    def save(self, path: str) -> None:
        """Save both backgrounds to file."""
        np.savez(path,
                 bg_day=(self._bgs[False] if self._bgs[False] is not None
                         else np.array([])),
                 bg_night=(self._bgs[True] if self._bgs[True] is not None
                           else np.array([])),
                 threshold=self.threshold, leak_every=self.leak_every,
                 open_kernel=self.open_kernel)

    def load_background(self, path: str) -> bool:
        """Restores the learned backgrounds."""
        if not os.path.isfile(path):
            return False
        data = np.load(path)
        for ctx, key in ((False, "bg_day"), (True, "bg_night")):
            if data[key].size:
                self._bgs[ctx] = data[key].copy()
        return True

    @classmethod
    def from_video(cls, path: str, n_samples: int = 60, threshold: float = 25,
                   percentile: float = 90,
                   seed: int | None = None) -> "BackgroundSubtractor":
        """Cheat: sample n random frames across a whole video to
        compute the background. Offline, so it can look at the bright tail of
        the whole recording at once instead of tracking an envelope."""
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_samples = min(n_samples, total)
        indices = np.random.default_rng(seed).choice(total, size=n_samples,
                                                     replace=False)
        frames = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            raise ValueError(f"couldn't read any frames from {path}")

        sub = cls(threshold=threshold)
        sub._bgs[False] = np.quantile(np.stack(frames), percentile / 100,
                                      axis=0, method="higher")
        return sub

    @staticmethod
    def _scale(frame: np.ndarray, gain: float) -> np.ndarray:
        return np.clip(frame.astype(np.float32) * gain, 0, 255
                       ).astype(np.uint8)


class CorridorDetection(CorridorDetectionBase):
    """New background subtraction algo for CORRIDOR areas. Instead of
    thresholding the raw frame, subtracts the 'learned' background and then
    applies the cv2.threshold.

    See doc above.

    Params:
        leak_every: int
        control_tolerance: float
        stuck_minutes: int
        diff_threshold: int  # FIXME delete
        diff_threshold_night: int  # FIXME delete
    Actions:
        Flush/Save/Delete the saved background with buttons in UI.
    """

    PARAMS = [
        CustomDetectionParam(
            "leak_every", int, LEAK_EVERY, "#Background leak (frames/level)",
            0, 6000,
            "How fast the background absorbs something that stays in the "
            "corridor: one grey level every N frames. Lower absorbs a real "
            "change (an object moved, a new shadow) faster, but also absorbs "
            "a sleeping animal sooner (@ 1800 and 10FPS, a real change clears "
            "in ~6 h and an animal asleep in the corridor is still detected "
            "after 3.5 h). 0 = never, manual flush only."),
        CustomDetectionParam(
            "control_tolerance", float, CONTROL_TOLERANCE,
            "#Scene-change tolerance", 0.0, 0.5,
            "Relative brightness change on CONTROL_ROI that counts as a "
            "lighting change and rescales the backgrounds. 0.02 = 2%. Lower "
            "tracks dimming more closely; too low rescales on noise."),
        CustomDetectionParam(
            "stuck_minutes", int, STUCK_MINUTES, "#Stuck-area alarm (min)",
            0, 240,
            "Telegram alarm when one area stays occupied this long without "
            "ever clearing --> an animal immobile in the corridor, or "
            "needs cleaning. To be kept well under the time it takes the "
            "background to absorb changes, so the alarm fires while the area "
            "is still being reported occupied. 0 = no alarm."),
        CustomDetectionParam(  # FIXME delete
            "diff_threshold", int, DIFF_THRESHOLD,
            "#New-method threshold (day)", 1, 255,
            "Temporary params for the new detection method. "
            "One value for all 4 areas."),
        CustomDetectionParam(  # FIXME delete
            "diff_threshold_night", int, DIFF_THRESHOLD_NIGHT,
            "#New-method threshold (night)", 1, 255,
            "Same for night.")]

    ACTIONS = ["flush", "save", "delete_save"]

    def __init__(self) -> None:
        super().__init__()
        self.subs: dict[int, BackgroundSubtractor] = {}

        for i in range(4):
            # placeholder, overwritten every frame from self.diff_threshold
            sub = BackgroundSubtractor(threshold=25)
            sub.load_background(self._save_path(i))
            self.subs[i] = sub

        # brightness of CONTROL_ROI when the backgrounds were last in sync
        self._control_levels: dict[bool, float | None] = {False: None,
                                                          True: None}
        self._load_control_levels()

        # continuous-occupancy timers, one per area
        self._occupied_since: list[float | None] = [None] * 4
        self._alarmed: list[bool] = [False] * 4
        # last seen geometry, so save() can write viewable background images
        self._areas: list[list[int]] = []
        self._frame_shape: tuple[int, ...] | None = None

    def detect(self, cam) -> None:
        # FIXME: SHADOW MODE: the previous detection still runs unchanged and
        # is what actually drives corridor doors/access logic. The new method
        # runs alongside for comparison only before we can trust it.
        super().detect(cam)  # FIXME: delete
        try:
            self._shadow_detect(cam)
        except Exception as e:
            if log is not None:
                log.error("corridor shadow detection failed: %s", e,
                          exception=traceback.format_exc())

    def _relevel_from_control(self, cam) -> None:
        """Corrects the backgrounds for a lighting change, measured
        from CONTROL_ROI. Runs every frame, so a slow dim is tracked as it
        happens and an abrupt flip is detected within a frame or two, and the
        per-pixel background is never discarded."""
        cx1, cy1, cx2, cy2 = CONTROL_ROI
        level = float(cam.gray_frame[cy1:cy2, cx1:cx2].mean())
        ref = self._control_levels[cam.night]
        if ref is None:  # first frame in this light context
            # Carry the other context's backgrounds over rather than letting
            # each area cold start (avoid case an animal is already in the
            # corridor at exactly 8AM/PM).
            other_ref = self._control_levels[not cam.night]
            if other_ref is not None:
                gain = level / max(other_ref, 1.0)
                for sub in self.subs.values():
                    sub.carry_over(gain, to_night=cam.night)
            self._control_levels[cam.night] = level
            return
        gain = level / max(ref, 1.0)
        if abs(gain - 1.0) > self.control_tolerance:
            for sub in self.subs.values():
                sub.relevel(gain, night=cam.night)
            self._control_levels[cam.night] = level

    def _check_stuck(self, cam, now: float | None = None) -> None:
        # FIXME: add failsafe mechanics once we trust the new detection.
        # TODO: maybe turn RFID off so no door moves if an animal is stuck.
        # TODO: maybe also repeat alarm
        """Alarms when an area has been occupied without clearing
        in STUCK_MINUTES."""
        if not self.stuck_minutes:
            return
        stuck = _stuck_areas(cam.counts, cam.zero_or_one_mouse,
                             self._occupied_since, self._alarmed,
                             self.stuck_minutes * 60,
                             time.monotonic() if now is None else now)
        for i in stuck:
            if log is not None:
                log.alarm(f"Corridor area {i + 1} has been occupied for "
                          f"{self.stuck_minutes} min without clearing. An "
                          f"animal may be immobile in the corridor, or "
                          f"something may have been left there.")

    def _shadow_detect(self, cam) -> None:
        self._relevel_from_control(cam)
        self._check_stuck(cam)
        self._areas = cam.areas
        self._frame_shape = cam.gray_frame.shape

        new_masks: dict[int, np.ndarray] = {}
        new_counts: dict[int, int] = {}
        for i, (x1, y1, x2, y2) in enumerate(cam.areas):
            if not cam.areas_active[i]:
                # cam.masks[i] = -1  # FIXME add
                # cam.counts[i] = -1  # FIXME add
                continue
            sub = self.subs.setdefault(i, BackgroundSubtractor())
            sub.threshold = (self.diff_threshold_night if cam.night  # FIXME
                             else self.diff_threshold)  # FIXME delete
            # sub.threshold = cam.thresholds[i]  # FIXME add
            sub.leak_every = int(self.leak_every)
            # which side of the pixel history the background is on
            sub.dark_subjects = Color is None or cam.color == Color.BLACK
            crop = cam.gray_frame[y1:y2, x1:x2]
            # cam.masks[i] = sub.apply(crop, night=cam.night)  # FIXME add
            # cam.counts[i] = cv2.countNonZero(cam.masks[i])  # FIXME add

            new_masks[i] = sub.apply(crop, night=cam.night)  # FIXME delete
            new_counts[i] = cv2.countNonZero(new_masks[i])  # FIXME delete

        self._log_shadow(cam, new_counts)  # FIXME delete
        cam.items_to_draw["shadow_new_masks"] = new_masks  # FIXME delete

    def _log_shadow(self, cam, new_counts: dict[int, int]) -> None:
        """Appends a row whenever the new method would have made a
        different empty/one/multiple call than the old detection."""
        rows = []
        for i, new_count in new_counts.items():
            old_count = cam.counts[i]
            if old_count < 0:
                continue  # area inactive
            old_class = _classify(old_count, cam.zero_or_one_mouse,
                                  cam.one_or_two_mice)
            new_class = _classify(new_count, cam.zero_or_one_mouse,
                                  cam.one_or_two_mice)
            if old_class == new_class:
                continue
            rows.append(f"{datetime.now().isoformat()},{i + 1},"
                        f"{old_count},{new_count},"
                        f"{old_class},{new_class}\n")
        if not rows:
            return
        path = self._shadow_log_path()
        write_header = not os.path.exists(path)
        header = "timestamp,area,old_count,new_count,old_class,new_class\n"
        with open(path, "a") as f:
            if write_header:
                f.write(header)
            f.writelines(rows)

    def _shadow_log_path(self) -> str:
        f = Path(settings.get("DATA_DIRECTORY")) / "corridor_shadow_log.csv"
        return str(f)

    def flush(self) -> None:
        for sub in self.subs.values():
            sub.flush()
        self._control_levels = {False: None, True: None}

    def save(self) -> None:
        for i, sub in self.subs.items():
            sub.save(self._save_path(i))
        np.savez(self._control_save_path(),
                 levels=np.array([np.nan if self._control_levels[c] is None
                                  else self._control_levels[c]
                                  for c in (False, True)]))
        self._save_background_images()

    def _save_background_images(self) -> None:
        """Save day/night backgrounds as images."""
        if self._frame_shape is None:
            return  # no bg yet
        for ctx, name in ((False, "day"), (True, "night")):
            canvas = np.zeros(self._frame_shape, np.uint8)
            drawn = False
            for i, (x1, y1, x2, y2) in enumerate(self._areas):
                bg = self.subs[i]._bgs[ctx] if i in self.subs else None
                if bg is not None and bg.shape == (y2 - y1, x2 - x1):
                    canvas[y1:y2, x1:x2] = bg
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, 1)
                    drawn = True
            if drawn:
                cv2.imwrite(self._image_path(name), canvas)

    def _image_path(self, name: str) -> str:
        fn = f"corridor_bg_{name}.png"
        return str(Path(settings.get("DATA_DIRECTORY")) / fn)

    def _load_control_levels(self) -> None:
        path = self._control_save_path()
        if not os.path.isfile(path):
            return
        data = np.load(path)
        if "levels" not in data:
            return
        for ctx, level in zip((False, True), data["levels"]):
            self._control_levels[ctx] = (None if np.isnan(level)
                                         else float(level))

    def delete_save(self) -> None:
        paths = [self._save_path(i) for i in range(4)]
        paths.append(self._control_save_path())
        paths += [self._image_path(n) for n in ("day", "night")]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

    def _control_save_path(self) -> str:
        fn = "corridor_bg_control.npz"
        return str(Path(settings.get("DATA_DIRECTORY")) / fn)

    def _save_path(self, i: int) -> str:
        fn = f"corridor_bg_area{i + 1}.npz"
        return str(Path(settings.get("DATA_DIRECTORY")) / fn)


def _classify(count: int, empty_limit: int, subject_limit: int) -> int:
    """0 = empty, 1 = one subject, 2 = multiple."""
    if count < empty_limit:
        return 0
    if count <= subject_limit:
        return 1
    return 2


def _stuck_areas(counts: list[int], limit: int, since: list[float | None],
                 alarmed: list[bool], timeout_s: float,
                 now: float) -> list[int]:
    """Which areas have just crossed `timeout_s` of continuous occupancy.

    Updates `since`/`alarmed` in place. Returns each area only once per
    episode; an area that clears resets and can alarm again later."""
    crossed = []
    for i, count in enumerate(counts):
        if count < limit:  # empty, or inactive (-1)
            since[i] = None
            alarmed[i] = False
        elif since[i] is None:
            since[i] = now
        elif not alarmed[i] and now - since[i] >= timeout_s:
            alarmed[i] = True
            crossed.append(i)
    return crossed


def demo() -> None:
    """Test: a mouse must never end up in the background, whatever it does."""
    rng = np.random.default_rng(0)
    mouse = (slice(10, 22), slice(25, 50))

    def floor(gain: float = 1.0) -> np.ndarray:
        base = np.full((30, 75), 180, np.uint8) + rng.integers(
            0, 4, (30, 75), dtype=np.uint8)
        return BackgroundSubtractor._scale(base, gain)

    def with_mouse(gain: float = 1.0, where=mouse) -> np.ndarray:
        f = floor(gain)
        f[where] = np.uint8(max(1, int(60 * gain)))  # dark mouse
        return f

    # 1. the background is bootstrapped from a frame containing a mouse.
    #    It must be repaired as soon as the mouse moves off.
    sub = BackgroundSubtractor(threshold=50)
    sub.apply(with_mouse())
    assert not sub.apply(floor()).any(), "ghost survived the mouse leaving"
    assert sub.apply(with_mouse()).any(), "repaired background is blind"

    # 2. an animal asleep in the corridor must still be detected hours later:
    #    at the shipped leak rate it must survive a 1 h nap untouched.
    sub = BackgroundSubtractor(threshold=50)
    sub.apply(floor())
    for _ in range(60 * 60 * 10):  # 1 h at 10 fps
        mask = sub.apply(with_mouse())
    assert mask.any(), "sleeping mouse was absorbed into the background"

    # 3. a light change while a mouse sits in the corridor: relevel must keep
    #    detecting it and must not report the whole ROI as foreground.
    sub = BackgroundSubtractor(threshold=50)
    for _ in range(30):
        sub.apply(floor())
    sub.relevel(1.3)
    assert sub.apply(with_mouse(1.3)).any(), "relevel lost the mouse"
    assert not sub.apply(floor(1.3)).any(), "relevel left false foreground"

    # 4. a real change to the corridor (something dark left behind) IS
    #    absorbed, on the leak's slow timescale, without a manual flush.
    sub = BackgroundSubtractor(threshold=50, leak_every=1)
    sub.apply(floor())
    blob = (slice(0, 5), slice(0, 5))
    for _ in range(255):
        mask = sub.apply(with_mouse(where=blob))
    assert not mask.any(), "leak never absorbed a permanent change"

    # 5. a saved background comes back usable: detecting from the very first
    #    frame, and writable in place (np.load hands back read-only arrays).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bg.npz")
        src = BackgroundSubtractor(threshold=50)
        src.apply(floor())
        src.save(path)
        sub = BackgroundSubtractor(threshold=50)
        assert sub.load_background(path), "background did not load"
        assert sub.apply(with_mouse()).any(), "restored background is blind"

    # 6. switching from day to night (8AM/8PM) with a mouse already
    #    inside: carrying the background across must keep detecting it, where
    #    cold-starting that context would have taken the mouse for corridor.
    sub = BackgroundSubtractor(threshold=40)
    for _ in range(30):
        sub.apply(floor())
    assert sub.carry_over(0.35, to_night=True), "carry_over did not seed"
    dim = with_mouse(0.35)
    assert sub.apply(dim, night=True).any(), "mouse lost at the light change"
    cold = BackgroundSubtractor(threshold=40)
    assert not cold.apply(dim, night=True).any(), "cold start should be blind"

    # 7. the stuck-area alarm: fires once per episode, only after the timeout,
    #    and rearms only once the area has actually cleared.
    since, alarmed = [None] * 4, [False] * 4
    args = (50, since, alarmed, 60.0)  # limit=50 px, timeout=60 s
    assert _stuck_areas([0, 0, 0, 0], *args, 0.0) == []
    assert _stuck_areas([900, 0, 0, -1], *args, 10.0) == []  # just arrived
    assert _stuck_areas([900, 0, 0, -1], *args, 50.0) == []  # not yet
    assert _stuck_areas([900, 0, 0, -1], *args, 80.0) == [0]  # crossed
    assert _stuck_areas([900, 0, 0, -1], *args, 999.0) == []  # only once
    assert _stuck_areas([0, 0, 0, -1], *args, 1000.0) == []  # cleared
    assert _stuck_areas([900, 0, 0, -1], *args, 1001.0) == []  # timer resets
    assert _stuck_areas([900, 0, 0, -1], *args, 1062.0) == [0]  # can refire

    # 8. resizing an area in the GUI must not make the detector
    # blind to the mouse.
    sub.apply(floor()[:20, :40])
    assert sub.apply(with_mouse()[:20, :40]).any(), "resized area is blind"

    print("all tests ok")


if __name__ == "__main__":
    demo()
