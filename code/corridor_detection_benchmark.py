import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from corridor_detection import (CONTROL_ROI, CONTROL_TOLERANCE,
                                OPEN_KERNEL, BackgroundSubtractor,
                                _classify)


def _original_detect(gray: np.ndarray, areas: list, thresholds: list,
                     black: bool) -> tuple[list[int], list[np.ndarray]]:
    """Reimplements the original per-area detection."""
    mode = cv2.THRESH_BINARY_INV if black else cv2.THRESH_BINARY
    counts, masks = [], []
    for (x1, y1, x2, y2), threshold in zip(areas, thresholds):
        roi = gray[y1:y2, x1:x2]
        _, mask = cv2.threshold(roi, threshold, 255, mode)
        counts.append(cv2.countNonZero(mask))
        masks.append(mask)
    return counts, masks


def _hot_start_backgrounds(video_path: str, areas: list, n_samples: int = 60,
                           seed: int | None = None,
                           center_frame: int | None = None,
                           window_frames: int | None = None
                           ) -> list[np.ndarray]:
    """build background with random frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if center_frame is not None and window_frames is not None:
        lo = max(0, center_frame - window_frames // 2)
        hi = min(total, center_frame + window_frames // 2)
    else:
        lo, hi = 0, total
    pool = hi - lo
    n_samples = min(n_samples, pool) if pool > 0 else n_samples
    indices = lo + np.random.default_rng(seed).choice(pool, size=n_samples,
                                                      replace=False)
    crops: list[list[np.ndarray]] = [[] for _ in areas]
    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, (x1, y1, x2, y2) in enumerate(areas):
            crops[i].append(gray[y1:y2, x1:x2])
    cap.release()

    q = 0.9 if BENCHMARK_BLACK else 0.1  # switch if White on Black
    return [np.quantile(np.stack(c), q, axis=0, method="higher")
            for c in crops]


def _paste_masks(areas: list, masks: list, shape: tuple) -> np.ndarray:
    """Composites per-area masks back into a full-frame-sized canvas
    (zeros elsewhere) so they can be shown/written as a normal frame."""
    canvas = np.zeros(shape, np.uint8)
    for (x1, y1, x2, y2), mask in zip(areas, masks):
        canvas[y1:y2, x1:x2] = mask
    return canvas


_BRIGHT_GREEN = (0, 255, 0)
_LIGHT_GREEN = (144, 238, 144)


def _draw_contours(frame: np.ndarray, areas: list, masks: list) -> np.ndarray:
    """Draws the detected blobs."""
    canvas = _paste_masks(areas, masks, frame.shape[:2])
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    cv2.drawContours(out, contours, -1, _LIGHT_GREEN, thickness=cv2.FILLED)
    cv2.drawContours(out, contours, -1, _BRIGHT_GREEN, thickness=2)
    return out


def _plot_comparison(out_path: str, frame_numbers: list[int],
                     old_classes: list[list[int]],
                     new_classes: list[list[int]]) -> None:
    """One step-plot per area: old vs new classification over time
    (0=empty, 1=one, 2=multiple), with a red band wherever they disagree."""

    n_areas = len(old_classes)
    fig, axes = plt.subplots(n_areas, 1, sharex=True,
                             figsize=(12, 2.2 * n_areas))
    if n_areas == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        old_y = old_classes[i]
        new_y = new_classes[i]
        mismatch = [o != n for o, n in zip(old_y, new_y)]
        ax.fill_between(frame_numbers, 0, 1, where=mismatch, step="post",
                        transform=ax.get_xaxis_transform(),
                        color="red", alpha=0.2, linewidth=0)
        ax.step(frame_numbers, old_y, where="post", label="old",
                color="tab:blue")
        ax.step(frame_numbers, new_y, where="post", label="new",
                color="tab:orange", linestyle="--")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["empty", "one", "multiple"])
        ax.set_ylabel(f"area {i + 1}")
        ax.set_ylim(-0.3, 2.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle("old vs new corridor detection (red = mismatch)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote comparison plot to {out_path}")


class _BenchmarkRunner:
    def __init__(self, night: bool,
                 writer: cv2.VideoWriter | None = None) -> None:
        self.night = night
        self.writer = writer
        self.areas = [a[0:4] for a in BENCHMARK_AREAS]
        self.thresholds = [a[5] if night else a[4] for a in BENCHMARK_AREAS]
        self.black = BENCHMARK_BLACK
        self.empty_limit = BENCHMARK_EMPTY_LIMIT
        self.subject_limit = BENCHMARK_SUBJECT_LIMIT

        self.subs = [BackgroundSubtractor(threshold=t, open_kernel=OPEN_KERNEL,
                                          dark_subjects=BENCHMARK_BLACK)
                     for t in NEW_BG_THRESHOLDS]

        # store prev brightness of CONTROL_ROI
        self.control_level: float | None = None
        self.relevels = 0

        self.total_frames = 0
        self.critical = 0
        self.abs_diffs: list[int] = []
        self.frame_numbers: list[int] = []
        self.old_classes: list[list[int]] = [[] for _ in self.areas]
        self.new_classes: list[list[int]] = [[] for _ in self.areas]

    def hot_start(self, video_path: str, center_frame: int,
                  window_frames: int) -> None:
        print("hot-starting backgrounds from random frames...")
        for sub, bg in zip(self.subs, _hot_start_backgrounds(
                video_path, self.areas, seed=0, center_frame=center_frame,
                window_frames=window_frames)):
            sub._bgs[self.night] = bg
        control_bg = _hot_start_backgrounds(
            video_path, [CONTROL_ROI], seed=0, center_frame=center_frame,
            window_frames=window_frames)[0]
        self.control_level = float(control_bg.mean())

    def step(self, frame: np.ndarray, frame_label: int) -> None:
        """Processes one frame."""
        night = self.night
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx1, cy1, cx2, cy2 = CONTROL_ROI
        level = float(gray[cy1:cy2, cx1:cx2].mean())
        releveled = False
        if self.control_level is None:
            self.control_level = level
        else:
            gain = level / max(self.control_level, 1.0)
            if abs(gain - 1.0) > CONTROL_TOLERANCE:
                for sub in self.subs:
                    sub.relevel(gain, night=night)
                self.control_level = level
                self.relevels += 1
                releveled = True

        old_counts, old_masks = _original_detect(gray, self.areas,
                                                 self.thresholds, self.black)
        new_masks = [self.subs[i].apply(gray[y1:y2, x1:x2], night=night)
                     for i, (x1, y1, x2, y2) in enumerate(self.areas)]
        new_counts = [cv2.countNonZero(m) for m in new_masks]

        if self.writer is not None:
            old_vis = _draw_contours(frame, self.areas, old_masks)
            new_vis = _draw_contours(frame, self.areas, new_masks)
            roi_color = (0, 0, 255) if releveled else (255, 0, 0)
            for panel in (frame, old_vis, new_vis):
                cv2.rectangle(panel, (cx1, cy1), (cx2, cy2), roi_color, 2)
            self.writer.write(np.hstack([frame, old_vis, new_vis]))

        self.total_frames += 1
        self.frame_numbers.append(frame_label)
        for i in range(len(self.areas)):
            c_old = _classify(old_counts[i], self.empty_limit,
                              self.subject_limit)
            c_new = _classify(new_counts[i], self.empty_limit,
                              self.subject_limit)
            self.old_classes[i].append(c_old)
            self.new_classes[i].append(c_new)
            self.abs_diffs.append(abs(old_counts[i] - new_counts[i]))
            if c_old != c_new:
                self.critical += 1
                print(f"frame {frame_label} area{i + 1}: old={c_old}"
                      f"({old_counts[i]}) new={c_new}({new_counts[i]})")

    def report(self, plot_out_path: str) -> None:
        print(f"\n{self.total_frames} frames compared")
        if self.abs_diffs:
            print(f"mean |count diff|: {np.mean(self.abs_diffs):.1f}")
        print(f"critical classification mismatches: {self.critical}")
        print(f"relevels (lighting changes tracked): {self.relevels}")
        _plot_comparison(plot_out_path, self.frame_numbers, self.old_classes,
                         self.new_classes)


def _benchmark(video_path: str, start_s: float, duration_s: float,
               night: bool, save_video: bool = False,
               hot_start: bool = False) -> None:

    assert os.path.exists(video_path), f"video not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    video_out_path = video_path.rsplit(".", 1)[0] + "_benchmark.mp4"
    if save_video:
        writer = cv2.VideoWriter(video_out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w * 3, h))

    runner = _BenchmarkRunner(night, writer)
    if hot_start:
        runner.hot_start(video_path, center_frame=int(start_s * fps),
                         window_frames=int(120 * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) + duration_s * fps

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        runner.step(frame, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote side-by-side video to {video_out_path}")

    plot_out_path = video_path.rsplit(".", 1)[0] + "_benchmark_plot.png"
    runner.report(plot_out_path)


def _list_videos(folder: str) -> list[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(".mp4"))


def _benchmark_folder(folder_path: str, night: bool, save_video: bool = False,
                      hot_start: bool = False) -> None:
    """Same comparison as _benchmark, but runs every video file in folder."""
    video_paths = _list_videos(folder_path)
    if not video_paths:
        raise ValueError(f"no .mp4 files found in {folder_path}")
    print(f"found {len(video_paths)} videos in {folder_path}:")
    for p in video_paths:
        print(f"  {os.path.basename(p)}")

    probe = cv2.VideoCapture(video_paths[0])
    fps = probe.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    writer = None
    video_out_path = os.path.join(folder_path, "tracking_benchmark_output.mp4")
    if save_video:
        writer = cv2.VideoWriter(video_out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w * 3, h))

    runner = _BenchmarkRunner(night, writer)
    if hot_start:
        runner.hot_start(video_paths[0], center_frame=0,
                         window_frames=int(120 * fps))

    global_frame = 0
    for video_path in video_paths:
        print(f"processing {os.path.basename(video_path)}...")
        cap = cv2.VideoCapture(video_path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            global_frame += 1
            runner.step(frame, global_frame)
        cap.release()

    if writer is not None:
        writer.release()
        print(f"wrote side-by-side video to {video_out_path}")

    plot_out_path = os.path.join(folder_path, "tracking_benchmark_plot.png")
    runner.report(plot_out_path)


if __name__ == "__main__":
    BENCHMARK_AREAS = [  # from village
        (434, 230, 573, 318, 50, 40),   # AREA1_CORRIDOR
        (272, 288, 430, 314, 65, 75),   # AREA2_CORRIDOR
        (195, 288, 270, 318, 65, 80),   # AREA3_CORRIDOR
        (43, 290, 185, 317, 60, 65),    # AREA4_CORRIDOR
    ]
    BENCHMARK_EMPTY_LIMIT = 50
    BENCHMARK_SUBJECT_LIMIT = 2600
    BENCHMARK_BLACK = True

    # FIXME: hardcoded offline. will be read from UI (cam.thresholds[i]).
    NEW_BG_THRESHOLDS = [50] * 4

    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _path = _args[0]
    _kwargs = dict(night="--night" in sys.argv,
                   save_video="--video" in sys.argv,
                   hot_start="--hot-start" in sys.argv)
    if os.path.isdir(_path):
        _benchmark_folder(_path, **_kwargs)
    else:
        _benchmark(_path,
                   start_s=float(_args[1]) if len(_args) > 1 else 0.0,
                   duration_s=float(_args[2]) if len(_args) > 2 else 180.0,
                   **_kwargs)
