import numpy as np
from scipy.spatial.transform import Rotation as R

def rigid_transform_3D(A, B):
    """
    Solve R, t that aligns A to B:  B = R*A + t
    A: Nx3  (camera coordinates)
    B: Nx3  (world coordinates)
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    t = centroid_B - R_mat @ centroid_A
    return R_mat, t


def convert_pose(Pw_all_frames, timestamps, save_path):
    """
    Convert the world coordinates to camera coordinates.
    Pw_all_frames: (N, 4, 3), world coordinates of 4 points at each frame
    timestamps: (N,), timestamps of each pose frame
    """

    # --- Step 1: Build camera local coordinates using frame 0 ---
    Pw0 = Pw_all_frames[0]  # shape (4, 3), world coords of 4 points at frame 0
    print("Pw0:", Pw0)
    # Pc = Pw0 - Pw0[0]       # subtract point 1 → camera coordinate system
    Pc = Pw0
    print("Pc:", Pc)

    # --- Step 2: Process each frame ---
    poses = []
    for Pw, timestamp in zip(Pw_all_frames, timestamps):  # Pw shape = (4, 3)
        R_mat, t = rigid_transform_3D(Pc, Pw)
        quat = R.from_matrix(R_mat).as_quat()  # (qx, qy, qz, qw)
        poses.append(np.concatenate([t, quat]))

    # --- Step 3: Print the poses ---
    with open(save_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, pose in zip(timestamps, poses):
            t = pose[:3]
            q = pose[3:]
            f.write(f"{timestamp} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


def load_points_from_trc(trc_file):
    """
    trc file:
     - first 5 lines are header
     - from line 6, each line:
       - '<frame#>\t<timestamp>\t<x1>\t<y1>\t<z1>\t<x2>\t<y2>\t<z2>\t<x3>\t<y3>\t<z3>\t<x4>\t<y4>\t<z4>'
    
    return:
        - points: (N, 4, 3), world coordinates of 4 points at each frame, unit: mm
        - timestamps: (N,), timestamps of each pose frame, unit: s
    """
    with open(trc_file, "r") as f:
        lines = f.readlines()
    points = []
    timestamps = []
    for line in lines[5:]:
        frame, timestamp, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 = line.split()
        points.append([
            [float(x1), float(y1), float(z1)],
            [float(x2), float(y2), float(z2)],
            [float(x3), float(y3), float(z3)],
            [float(x4), float(y4), float(z4)]
        ])
        timestamps.append(float(timestamp))
    return np.array(points, dtype=np.float32), np.array(timestamps, dtype=np.float32)

if __name__ == "__main__":

    file_name = "20250804-3"
    trc_file = f"E:\\20250804_data\\20250804 pose raw data\\{file_name}.trc"
    save_path = f"{file_name}_tumformat.txt"

    Pw_all_frames, timestamps = load_points_from_trc(trc_file)
    print("Loaded points and timestamps")
    print(Pw_all_frames.shape, timestamps.shape)
    print(Pw_all_frames[0], timestamps[0])
    print()

    # convert the unit of points from mm to m
    Pw_all_frames = Pw_all_frames / 1000.
    print("Converted points to meters")
    print(Pw_all_frames[0], timestamps[0])
    print()

    convert_pose(Pw_all_frames, timestamps, save_path)
    print(f"Converted pose and saved to {save_path}")
