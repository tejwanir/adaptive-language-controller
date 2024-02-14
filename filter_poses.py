import json
from pathlib import Path
from typing import Generic, Iterable, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3D
from tqdm import tqdm

# This is just for plotting
JOINT_CONNECTIONS = [
    ("Head", "TopSkull"),
    ("TopSkull", "BackSkull"),
    ("Head", "Nose"),
    ("Nose", "EyeLeft"),
    ("Nose", "EyeRight"),
    ("Head", "Neck"),
    ("Neck", "Chest"),
    ("Chest", "Pelvis"),
    ("Pelvis", "HipLeft"),
    ("Pelvis", "HipRight"),
    ("Chest", "ClavicleLeft"),
    ("ClavicleLeft", "ShoulderLeft"),
    ("ShoulderLeft", "ElbowLeft"),
    ("ElbowLeft", "WristLeft"),
    ("Chest", "ClavicleRight"),
    ("ClavicleRight", "ShoulderRight"),
    ("ShoulderRight", "ElbowRight"),
    ("ElbowRight", "WristRight"),
    ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"),
    ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"),
]


def get_session_base_path(session_id: int) -> Path:
    return Path(f"./Lab Data/V4 LightBuzz (Table)/lightbuzz_table_{session_id + 1}")


def fit_perspective_parameters(z_adjust: bool = False):
    # Numpy-based approach
    pos_3d, pos_2d = [], []
    for session in range(6):
        path = get_session_base_path(session)
        with open(path / "poses.jsonl", "r") as f:
            for skeleton in tqdm(json.loads(line) for line in f.readlines()):
                for skeleton in skeleton["skeletons"]:
                    for joint, joint_info in skeleton.items():
                        if joint == "user_id":
                            continue  # Not really a joint
                        if joint_info["confidence"] < 0.8:
                            continue  # Not confident in the joint
                        pos_3d.append(joint_info["pos3D"])
                        pos_2d.append(joint_info["pos2D"])
    pos_3d = np.array(pos_3d)
    pos_2d = np.array(pos_2d)
    far_apart = pos_3d[:, 2] > 0.1  # Don't screw up numerical stability
    pos_3d = pos_3d[far_apart]
    pos_2d = pos_2d[far_apart]
    offset_2d = np.array([320, 240])  # The camera is 640x480
    pos_2d -= offset_2d

    # Fit the perspective parameters with two possible models.
    # x_p, y_p: projected 2D coordinates
    # x, y, z: 3D coordinates
    if z_adjust:
        # The first model assumes that the camera is at (0, 0, -a)
        # x_p = k * x / (z + a) => x_p * z = k * x - a * x_p
        # y_p = k * y / (z + a) => y_p * z = k * y - a * y_p
        b = np.r_[pos_2d[:, 0] * pos_3d[:, 2], pos_2d[:, 1] * pos_3d[:, 2]]
        A = np.c_[
            np.r_[pos_3d[:, 0], pos_3d[:, 1]],
            -np.r_[pos_2d[:, 0], pos_2d[:, 1]],
        ]
        k, a = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"PERSPECTIVE_SCALING = {k}")
        print(f"PERSPECTIVE_DELTA_Z = {a}")
        plt.subplot(1, 2, 1)
        plt.scatter(A[:, 0], b)
        plt.scatter(A[:, 0], A[:, 0] * k + a * A[:, 1], color="red")
        plt.subplot(1, 2, 2)
        plt.scatter(A[:, 1], b)
        plt.scatter(A[:, 1], A[:, 0] * k + a * A[:, 1], color="red")
        plt.tight_layout()
        plt.show()
    else:
        # The second model assumes that the camera is at (0, 0, 0)
        # Now this seems to be the ground truth, based on LightBuzz's documentation at
        # https://lightbuzz.com/docs/general-information/2d-3d-skeleton-coordinates/
        # x_p = k * x / z
        # y_p = k * y / z
        b = np.r_[pos_2d[:, 0], pos_2d[:, 1]]
        A = np.r_[pos_3d[:, 0] / pos_3d[:, 2], pos_3d[:, 1] / pos_3d[:, 2]]
        k = np.linalg.lstsq(A.reshape(-1, 1), b, rcond=None)[0]
        print(f"PERSPECTIVE_SCALING = {k}")
        print(f"PERSPECTIVE_DELTA_Z = 0")
        plt.scatter(A, b)
        plt.plot(A, A * k, color="red")
        plt.show()


# Run fit_perspective_parameters(False) to get the perspective parameters
# Projection model:
# projected x = PERSPECTIVE_SCALING * x / (z + PERSPECTIVE_DELTA_Z)
# projected y = PERSPECTIVE_SCALING * y / (z + PERSPECTIVE_DELTA_Z)
PERSPECTIVE_SCALING = 637.71263687
PERSPECTIVE_DELTA_Z = 0


UserPose = dict[str, np.ndarray]
UserPoses = dict[int, UserPose]


def preprocess_poses(
    path: Path, confidence_threshold: float = 0.5
) -> Iterable[tuple[float, UserPoses]]:
    with open(path, "r") as f:
        for line in f.readlines():
            frame = json.loads(line)
            timestamp = frame["timestamp"]
            skeletons = frame["skeletons"]
            yield (
                timestamp,
                {
                    skeleton["user_id"]: {
                        joint: np.array(joint_info["pos3D"])
                        for joint, joint_info in skeleton.items()
                        if joint != "user_id"
                        and joint_info["confidence"] >= confidence_threshold
                    }
                    for skeleton in skeletons
                },
            )


def plot_session(session: int):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    timestamps, all_poses = list(
        zip(*preprocess_poses(get_session_base_path(session) / "poses.jsonl"))
    )
    timestamps: list[float]
    all_poses: list[UserPoses]
    pose_filter = PoseFilter()
    all_poses = [
        pose_filter.filter_skeletons(timestamp, poses)
        for timestamp, poses in zip(timestamps, all_poses)
    ]

    max_user_id = max(
        user_id for user_poses in all_poses for user_id in user_poses.keys()
    )
    user_joints: list[list[dict[str, np.ndarray]]] = []
    min_pos_3d = np.array([100] * 3)
    max_pos_3d = -min_pos_3d
    for user_poses in all_poses:
        cur_user_joints = []
        for _ in range(max_user_id + 1):
            cur_user_joints.append({})
        for user_id, pose in user_poses.items():
            for joint, pos in pose.items():
                min_pos_3d = np.minimum(min_pos_3d, pos)
                max_pos_3d = np.maximum(max_pos_3d, pos)
                cur_user_joints[user_id][joint] = pos
        user_joints.append(cur_user_joints)
    timestamps = np.array(timestamps)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, (1, 3), projection="3d")
    x_spread, y_spread, z_spread = max_pos_3d - min_pos_3d
    t = 0.1
    ax.set_xlim(min_pos_3d[0] - t * x_spread, max_pos_3d[0] + t * x_spread)
    ax.set_ylim(min_pos_3d[1] - t * y_spread, max_pos_3d[1] + t * y_spread)
    ax.set_zlim(min_pos_3d[2] - t * z_spread, max_pos_3d[2] + t * z_spread)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-90, azim=-90)

    def init_per_user_list():
        ret = []
        for _ in range(max_user_id + 1):
            ret.append([])
        return ret

    update_timestamp = init_per_user_list()
    ax_depth = fig.add_subplot(222)
    ax_depth.set_ylabel("Median perceived depth")
    ax_depth.set_xlabel("Time (s)")
    ax_depth.set_xlim(timestamps[0], timestamps[-1])
    ax_depth.set_ylim(0, 2)
    depth_data = init_per_user_list()

    ax_n_joints = fig.add_subplot(224)
    ax_n_joints.set_ylabel("Number of joints")
    ax_n_joints.set_xlabel("Time (s)")
    ax_n_joints.set_xlim(timestamps[0], timestamps[-1])
    ax_n_joints.set_ylim(0, 20)
    n_joints_data = init_per_user_list()

    bones: list[list[Line3D]] = []
    depths: list[Line2D] = []
    n_joints: list[Line2D] = []
    legend_handles = []
    for user_id in range(max_user_id + 1):
        bones.append([])
        for _ in range(len(JOINT_CONNECTIONS)):
            (artist,) = ax.plot([], [], [], color=color_cycle[user_id])
            bones[user_id].append(artist)
        depths.append(ax_depth.plot([], [], label=f"{user_id}")[0])
        n_joints.append(ax_n_joints.plot([], [], label=f"{user_id}")[0])
        legend_handles.append(
            Patch(color=color_cycle[user_id], label=f"User {user_id}")
        )
    ax.legend(handles=legend_handles)
    ax_depth.legend()
    ax_n_joints.legend()

    interval = 50
    n_frames = int((timestamps[-1] - timestamps[0]) * 1000 // interval)

    def update(frame: int):
        ts = timestamps[0] + frame * interval / 1000
        idx = np.searchsorted(timestamps, ts)
        affected_artists = []
        for user_id in range(max_user_id + 1):
            joints = user_joints[idx][user_id]
            for i, (joint1, joint2) in enumerate(JOINT_CONNECTIONS):
                artist = bones[user_id][i]
                affected_artists.append(artist)
                if (
                    joint1 not in user_joints[idx][user_id]
                    or joint2 not in user_joints[idx][user_id]
                ):
                    artist.set_data_3d([], [], [])
                    continue
                joint1_pos = joints[joint1]
                joint2_pos = joints[joint2]
                artist.set_data_3d(
                    [joint1_pos[0], joint2_pos[0]],
                    [joint1_pos[1], joint2_pos[1]],
                    [joint1_pos[2], joint2_pos[2]],
                )
            update_timestamp[user_id].append(ts)
            median_depth = np.median([joint[2] for joint in joints.values()])

            depth_data[user_id].append(median_depth + PERSPECTIVE_DELTA_Z)
            n_joints_data[user_id].append(len(joints))

            depths[user_id].set_data(update_timestamp[user_id], depth_data[user_id])
            n_joints[user_id].set_data(
                update_timestamp[user_id], n_joints_data[user_id]
            )

    plt.ion()
    for i in range(n_frames):
        update(i)
        plt.pause(interval / 1000)


T = TypeVar("T")


class LowPassFilter(Generic[T]):
    """A simple low-pass filter with a time constant omega.

    Transfer function: H(s) = omega / (s + omega)
    """

    def __init__(self, omega: float, initial_value: T):
        self.omega = omega
        self.last_timestamp = float("-inf")
        self.last_y = initial_value
        self.last_x = initial_value

    def __call__(self, timestamp: int, new_x: T) -> T:
        """Updates the filter with a new value and returns the filtered value.

        Assumes FOH interpolation between the last timestamp and the current one.
        """
        delta_t = timestamp - self.last_timestamp
        delta_x = self.last_x - new_x
        e = np.exp(-delta_t * self.omega)
        new_y = (
            new_x
            + (1 / delta_t / self.omega) * (1 - e) * delta_x
            + e * (self.last_y - self.last_x)
        )
        self.last_timestamp = timestamp
        self.last_x = new_x
        self.last_y = new_y
        return new_y


FILTER_ABOVE = lambda hz: 2 * hz * np.pi
OMEGA_JOINT = FILTER_ABOVE(2)
OMEGA_DEPTH = FILTER_ABOVE(1 / 30)


class PoseFilter:
    """A pose filter that filters the poses and corrects for depth drifting."""

    pose_filters: dict[int, dict[str, LowPassFilter[np.ndarray]]]
    depth_filters: dict[int, LowPassFilter[float]]

    def __init__(self):
        self.pose_filters = {}
        self.depth_filters = {}

    def filter_skeletons(self, timestamp: int, poses: UserPoses) -> UserPoses:
        low_pass_filtered = {}

        # Step 1: low-pass filter the poses
        for user_id, pose in poses.items():
            if user_id not in self.pose_filters:
                self.pose_filters[user_id] = {}
            last_pose = self.pose_filters[user_id]
            low_pass_filtered[user_id] = {}
            for joint, pos in pose.items():
                if joint not in last_pose:
                    last_pose[joint] = LowPassFilter(OMEGA_JOINT, pos)
                low_pass_filtered[user_id][joint] = last_pose[joint](timestamp, pos)

        # Step 2: inject strong prior of constant depth
        depth_corrected = {}
        for user_id, pose in low_pass_filtered.items():
            median_depth = np.median([pos[2] for pos in pose.values()])
            median_depth += PERSPECTIVE_DELTA_Z
            if user_id not in self.depth_filters:
                self.depth_filters[user_id] = LowPassFilter(OMEGA_DEPTH, median_depth)
            corrected_depth = self.depth_filters[user_id](timestamp, median_depth)
            correction = corrected_depth - median_depth
            depth_corrected[user_id] = {}
            for joint, pos in pose.items():
                joint_depth = pos[2] + PERSPECTIVE_DELTA_Z
                corrected_joint_depth = joint_depth + correction
                depth_corrected[user_id][joint] = np.array(
                    [
                        pos[0] * (corrected_joint_depth / joint_depth),
                        pos[1] * (corrected_joint_depth / joint_depth),
                        pos[2] + correction,
                    ]
                )

        return depth_corrected


if __name__ == "__main__":
    # fit_perspective_parameters(False)
    plot_session(0)
