import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

IMAGE_FOLDER = "/home/thomas/Documents/Village_Towers_Task/data/calib*.png"

DOT_SPACING = 15
WIDTH = 60


def build_t_shape(add_jitter=False):
    rng = np.random.default_rng(42)
    jitter = DOT_SPACING/8
    pts = []

    # horizontal line (centered at 0)
    for i in range(0, 641, DOT_SPACING):
        for j in range(-WIDTH//2, WIDTH//2 + 1, DOT_SPACING):
            x = i
            y = j + 210
            if add_jitter:
                x += rng.uniform(-jitter, jitter)
                y += rng.uniform(-jitter, jitter)
            pts.append([x, y, 0])

    # vertical line (from center upward)
    for j in range(0, 421, DOT_SPACING):
        for i in range(-WIDTH//2, WIDTH//2 + 1, DOT_SPACING):
            x = i
            y = j
            if add_jitter:
                x += rng.uniform(-jitter, jitter)
                y += rng.uniform(-jitter, jitter)
            pts.append([x, y, 0])

    return np.array(pts, dtype=np.float32)


OBJ_POINTS_TEMPLATE = build_t_shape()
TEST = build_t_shape(add_jitter=True)
TEST = TEST[:, :2]

def simple_square_template(w, h):
    pts = []
    for i in range(w):
        for j in range(h):
            pts.append([i * DOT_SPACING, j * DOT_SPACING, 0])
    return np.array(pts, dtype=np.float32)

# OBJ_POINTS_TEMPLATE = simple_square_template(10, 10)



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


def detect_dots(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints = detector.detect(gray)
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    # print(f"Detected {len(pts)} blobs")
    # print(pts)

    return pts


def match_points(image_pts, obj_pts):
    if len(image_pts) < 4:
        return None, None
    print(f"Matching {len(image_pts)} detected to {len(obj_pts)} model")

    img_center = np.mean(image_pts, axis=0)
    obj_center = np.mean(obj_pts[:, :2], axis=0)

    img_pts_norm = image_pts - img_center
    obj_pts_norm = obj_pts[:, :2] - obj_center

    matched_img = []
    matched_obj = []

    used = set()

    for op in obj_pts_norm:
        dists = np.linalg.norm(img_pts_norm - op, axis=1)
        idx = np.argmin(dists)

        if idx not in used:
            matched_img.append(image_pts[idx])
            matched_obj.append(obj_pts[np.where((obj_pts_norm == op).all(axis=1))[0][0]])
            used.add(idx)

    fig, ax = plt.subplots()
    ax.scatter(image_pts[:, 0], image_pts[:, 1], color='blue', label='Detected')
    ax.scatter(obj_pts[:, 0], obj_pts[:, 1], color='red', label='Model')
    for mi, mo in zip(matched_img, matched_obj):
        ax.plot([mi[0], mo[0]], [mi[1], mo[1]], color='gray', linestyle='--')
    ax.legend()
    ax.set_title("Matched Points")
    plt.show()

    if len(matched_img) < 4:
        return None, None

    return np.array(matched_obj, dtype=np.float32), np.array(matched_img, dtype=np.float32)

def calibrate():
    detector = create_blob_detector()

    objpoints = []
    imgpoints = []

    images = glob.glob(IMAGE_FOLDER)
    print(f"Found {len(images)} calibration images")

    for fname in images:
        if 'detected' in fname:
            continue
        img = cv2.imread(fname)

        pts = detect_dots(img, detector)

        for pt in pts:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), 2)
        cv2.imwrite(fname.replace(".png", "_detected.png"), img)

        print("HERE")
        obj, imgp = match_points(pts, OBJ_POINTS_TEMPLATE)
        obj, imgp = match_points(TEST, OBJ_POINTS_TEMPLATE)

        if obj is not None:
            objpoints.append(obj)
            imgpoints.append(imgp)

            print(f"[OK] {fname} -> {len(obj)} points")
        else:
            print(f"[SKIP] {fname}")

    if len(objpoints) < 4:
        raise RuntimeError("Not enough valid calibration images")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img.shape[:2][::-1],
        None,
        None
    )

    print("\n=== RESULTS ===")
    print("Reprojection error:", ret)
    print("Camera matrix:\n", K)
    print("Distortion:\n", dist)

    return K, dist

def undistort_image(img, K, dist):
    h, w = img.shape[:2]

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, K, dist, None, newK)

    return dst


def undistort_points(points, K, dist):
    """
    points: Nx2 array of (x,y)
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    undistorted = cv2.undistortPoints(pts, K, dist, P=K)

    return undistorted.reshape(-1, 2)


if __name__ == "__main__":
    import time 
    K, dist = calibrate()

    test = "/home/thomas/Documents/Village_Towers_Task/data/calib_0000.png"
    test_img = cv2.imread(test)
    undist = undistort_image(test_img, K, dist)
    cv2.imwrite("undistorted.jpg", undist)

    traj = np.array([[100, 200], [120, 210], [140, 220]])
    traj_undist = undistort_points(traj, K, dist)

    start = time.time()
    for _ in range(1000):
        undistort_points(traj, K, dist)
    end = time.time()
    print(f"Undistorting 1000 trajectories took {end - start:.3f} seconds")
    print("Original:", traj)
    print("Undistorted:", traj_undist)
