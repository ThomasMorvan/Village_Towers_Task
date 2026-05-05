import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FOLDER = "/home/thomas/Documents/Village_Towers_Task/data/calib*.png"

GRID_SIZE = (30, 18)
DOT_SPACING = 17

OBJ_POINTS_TEMPLATE = np.zeros((GRID_SIZE[0] * GRID_SIZE[1], 3), np.float32)
OBJ_POINTS_TEMPLATE[:, :2] = np.mgrid[0:GRID_SIZE[0], 0:GRID_SIZE[1]].T.reshape(-1, 2) * DOT_SPACING


def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.85
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 0
    return cv2.SimpleBlobDetector_create(params)


def calibrate(fnames):
    if isinstance(fnames, str):
        fnames = [fnames]

    detector = create_blob_detector()
    objpoints = []
    imgpoints = []
    gray = None

    for fname in fnames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findCirclesGrid(
            gray, GRID_SIZE,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=detector
        )

        if found:
            objpoints.append(OBJ_POINTS_TEMPLATE.copy())
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, GRID_SIZE, corners, found)
            cv2.imwrite(fname.replace(".png", "_detected.png"), img)
            print(f"[OK] {fname}")
        else:
            print(f"[SKIP] {fname}")

    if not objpoints:
        raise RuntimeError("No valid calibration frames found")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"\n=== RESULTS ({len(objpoints)}/{len(fnames)} frames used) ===")
    print("Reprojection error:", ret)
    print("Camera matrix:\n", K)
    print("Distortion:\n", dist)

    return K, dist, objpoints, imgpoints, rvecs, tvecs



def _score_line(name, value, unit, good, ok, fmt=".2f"):
    tag = "GOOD" if value <= good else ("MEH  " if value <= ok else "BAD")
    print(f"  {tag}  {name}: {value:{fmt}}{unit}  (good < {good}, ok < {ok})")


def alignment_report(img_path, K, dist, detector, objpoints, imgpoints, rvecs, tvecs):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    found, corners = cv2.findCirclesGrid(
        gray, GRID_SIZE, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector
    )
    if not found:
        print("Grid not detected in alignment_report")
        return

    _, rvec, tvec = cv2.solvePnP(OBJ_POINTS_TEMPLATE, corners, K, dist)

    proj, _ = cv2.projectPoints(OBJ_POINTS_TEMPLATE, rvec, tvec, K, dist)
    reproj_err = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1
    )))

    R, _ = cv2.Rodrigues(rvec)
    tilt_total = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
    tilt_x_deg = float(np.degrees(np.arcsin(np.clip(R[0, 2], -1.0, 1.0))))
    tilt_y_deg = float(np.degrees(np.arcsin(np.clip(R[1, 2], -1.0, 1.0))))

    pts = corners.reshape(-1, 2)
    row0 = pts[:GRID_SIZE[0]]
    dx, dy = row0[-1] - row0[0]
    roll_rad = float(np.arctan2(dy, dx))
    roll_deg = float(np.degrees(roll_rad))

    d = dist.ravel()
    k1 = float(d[0])
    tang = float(np.sqrt(d[2] ** 2 + d[3] ** 2))
    cx_off = float(K[0, 2] - w / 2)
    cy_off = float(K[1, 2] - h / 2)

    print(f"\n=== ALIGNMENT REPORT: {img_path} ===")
    print("  -- Mounting --")
    _score_line("Tilt total",             tilt_total,       "°", good=2.0, ok=5.0)
    _score_line("  Tilt X (left/right)",  abs(tilt_x_deg),  "°", good=2.0, ok=5.0)
    _score_line("  Tilt Y (fwd/back)",    abs(tilt_y_deg),  "°", good=2.0, ok=5.0)
    _score_line("Roll",                   abs(roll_deg),    "°", good=1.0, ok=3.0)
    _score_line("Pose reproj error",      reproj_err,      " px", good=0.5, ok=1.5)
    print("  -- Lens --")
    print(f"    k1={k1:+.4f} ({'pincushion' if k1>0 else 'barrel'})  "
          f"tang={tang:.5f}  principal=({cx_off:+.1f},{cy_off:+.1f})px")


    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Tilt X {tilt_x_deg:+.1f}°  Tilt Y {tilt_y_deg:+.1f}°  "
                 f"Roll {roll_deg:+.1f}°  Reproj {reproj_err:.2f}px")
    color_bar_args = dict(fraction=0.03, pad=0.03)

    # residual error
    ax = axes[0, 0]
    all_det, all_err = [], []
    for obj, imp, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        all_det.append(imp.reshape(-1, 2))
        all_err.append(p.reshape(-1, 2) - imp.reshape(-1, 2))
    all_pts  = np.concatenate(all_det)
    all_errs = np.concatenate(all_err)
    mags     = np.linalg.norm(all_errs, axis=1)

    tcf = ax.tricontourf(all_pts[:, 0], all_pts[:, 1], mags, levels=10, cmap="hot", alpha=0.2)
    q = ax.quiver(all_pts[:, 0], all_pts[:, 1],
                  all_errs[:, 0], all_errs[:, 1], mags,
                  cmap="hot", angles="xy", scale_units="xy", scale=0.05)
    fig.colorbar(q, ax=ax, label="Error (px)", **color_bar_args)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # intrinsic lens distortion
    ax = axes[0, 1]
    step = 40
    gx, gy = np.meshgrid(np.arange(step // 2, w, step), np.arange(step // 2, h, step))
    gpts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    upts = cv2.undistortPoints(gpts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    disp = upts - gpts
    dmag = np.linalg.norm(disp, axis=1)

    q2 = ax.quiver(gpts[:, 0], gpts[:, 1], disp[:, 0], disp[:, 1], dmag,
                   cmap="cool", angles="xy", scale_units="xy", scale=0.3)
    fig.colorbar(q2, ax=ax, label="Displacement (px)", **color_bar_args)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Tilt and roll
    ax = axes[1, 0]
    tl = pts[0];  tr = pts[GRID_SIZE[0] - 1]
    bl = pts[(GRID_SIZE[1] - 1) * GRID_SIZE[0]];  br = pts[-1]
    center = np.mean([tl, tr, bl, br], axis=0)
    avg_w  = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    avg_h  = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    hw, hh = avg_w / 2, avg_h / 2
    ideal = np.array([center + np.array([-cr*hw + sr*hh, -sr*hw - cr*hh]),
                      center + np.array([ cr*hw + sr*hh,  sr*hw - cr*hh]),
                      center + np.array([ cr*hw - sr*hh,  sr*hw + cr*hh]),
                      center + np.array([-cr*hw - sr*hh, -sr*hw + cr*hh])])

    ax.imshow(gray, cmap="gray")
    ax.scatter(pts[:, 0], pts[:, 1], s=2, c="cyan", zorder=3)
    ax.plot(*np.vstack([tl, tr, br, bl, tl]).T, "r-",  lw=2,
            label=f"Actual (tilt {tilt_total:.1f}°)")
    ax.plot(*np.vstack([ideal, ideal[0]]).T, "g--", lw=2, label="Ideal")
    ax.legend()
    ax.axis("off")

    # px / cm scale
    pts_grid = pts.reshape(GRID_SIZE[1], GRID_SIZE[0], 2)
    cm_per_dot = DOT_SPACING / 10.0

    scale_vals = []
    scale_pos  = []
    for i in range(GRID_SIZE[1]):
        for j in range(GRID_SIZE[0]):
            neighbours = []
            if j + 1 < GRID_SIZE[0]:
                neighbours.append(np.linalg.norm(pts_grid[i, j+1] - pts_grid[i, j]))
            if i + 1 < GRID_SIZE[1]:
                neighbours.append(np.linalg.norm(pts_grid[i+1, j] - pts_grid[i, j]))
            if neighbours:
                scale_vals.append(np.mean(neighbours) / cm_per_dot)
                scale_pos.append(pts_grid[i, j])

    scale_vals = np.array(scale_vals)
    scale_pos  = np.array(scale_pos)

    ax = axes[1, 1]
    tcf2 = ax.tricontourf(scale_pos[:, 0], scale_pos[:, 1], scale_vals,
                          levels=15, cmap="viridis")
    fig.colorbar(tcf2, ax=ax, label="px / cm", **color_bar_args)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

def print_explanations():
    metrics = ["Tilt", "Roll", "Pose reproj error", "k1", "Tangential", "Principal point offset"]
    explanations = [
        "Angle between camera and floor (trapezoid perspective)",
        "Rotation of image (diagonal grid rows)",
        "Quality of the pose fit (high == unreliable)",
        "Barrel (+) or pincushion (-) lens distortion",
        "Lens/sensor misalignment",
        "How far the optical center is from image center"
    ]
    how_to_fix = [
        "Tilt the camera",
        "Rotate camera",
        "Improve lighting or focus",
        "Lens property, correct in software",
        "Lens property, correct in software",
        "Lens property, correct in software",
    ]
    print("Metric explanations:")
    for m, e, f in zip(metrics, explanations, how_to_fix):
        print(f"- {m:25} {e:60} {f}")


def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, newK)


def undistort_points(points, K, dist):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def _perspective_homography(src_pts):
    pts_grid = src_pts.reshape(GRID_SIZE[1], GRID_SIZE[0], 2)
    h_spacing = np.mean(np.linalg.norm(np.diff(pts_grid, axis=1), axis=2))
    v_spacing = np.mean(np.linalg.norm(np.diff(pts_grid, axis=0), axis=2))
    centroid  = src_pts.mean(axis=0)

    cols = (np.arange(GRID_SIZE[0]) - (GRID_SIZE[0] - 1) / 2) * h_spacing + centroid[0]
    rows = (np.arange(GRID_SIZE[1]) - (GRID_SIZE[1] - 1) / 2) * v_spacing + centroid[1]
    gx, gy = np.meshgrid(cols, rows)
    dst_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H


def show_corrections(img_path, K, dist, detector):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    found, corners = cv2.findCirclesGrid(gray, GRID_SIZE,
                                         flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                                         blobDetector=detector)
    if not found:
        print("Grid not detected in show_corrections")
        return

    src_pts = corners.reshape(-1, 2).astype(np.float32)

    # lens only
    lens_fix = cv2.undistort(img, K, dist)

    # tilt+roll only
    H_raw = _perspective_homography(src_pts)
    tilt_fix = cv2.warpPerspective(img, H_raw, (w, h))

    # tilt+roll+ lens
    src_undist = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2).astype(np.float32)
    H_undist = _perspective_homography(src_undist)
    full_fix = cv2.warpPerspective(lens_fix, H_undist, (w, h))

    titles = ["Raw", "Lens only (K + dist)", "Tilt + roll only", "Full fix (lens + tilt + roll)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, title, im in zip(axes.ravel(), titles, [img, lens_fix, tilt_fix, full_fix]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        new_path = img_path.replace(".png", f"_{title.replace(' ', '_')}.png")
        cv2.imwrite(new_path, im)
        ax.set_title(title)
        ax.axis("off")
        print(f"\n=== {title} ===")
        check_correction(new_path, K, dist, detector)

    plt.tight_layout()
    plt.show()

def check_correction(img_path, K, dist, detector):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findCirclesGrid(gray, GRID_SIZE,
                                         flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                                         blobDetector=detector)
    if not found:
        print("Grid not detected in check_correction")
        return
    pts_grid = corners.reshape(GRID_SIZE[1], GRID_SIZE[0], 2)
    h_dists  = np.linalg.norm(np.diff(pts_grid, axis=1), axis=2)
    v_dists  = np.linalg.norm(np.diff(pts_grid, axis=0), axis=2)
    print(f"Horizontal spacing: {h_dists.mean():.1f} ± {h_dists.std():.2f} px")
    print(f"Vertical   spacing: {v_dists.mean():.1f} ± {v_dists.std():.2f} px")

if __name__ == "__main__":
    fnames = [f"/home/thomas/Documents/Village_Towers_Task/data/calib_000{i}.png" for i in range(3)]
    align_img = fnames[0]
    K, dist, objpoints, imgpoints, rvecs, tvecs = calibrate(fnames)

    detector = create_blob_detector()
    alignment_report(align_img, K, dist, detector, objpoints, imgpoints, rvecs, tvecs)
    show_corrections(align_img, K, dist, detector)
    print_explanations()


    bench = False
    if bench == True:
        import time

        test_img = cv2.imread(align_img)
        N = 1000
        start = time.time()
        for _ in range(N):
            undistort_image(test_img, K, dist)
        end = time.time()
        print(f"Undistorting {N} images took {end - start:.3f} seconds. "
            f"That's {(end - start)/N*1000:.3f} ms/image")

        traj = np.array([[100, 200], [120, 210], [140, 220]])
        traj_undist = undistort_points(traj, K, dist)

        start = time.time()
        for _ in range(N):
            undistort_points(traj, K, dist)
        end = time.time()
        print(f"Undistorting {N} trajectories took {end - start:.3f} seconds. "
            f"That's {(end - start)/N*1000:.3f} ms/trajectory")
        print("Original:", traj)
        print("Undistorted:", traj_undist)
