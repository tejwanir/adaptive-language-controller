import json
import time
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from record import JOINT_NAMES


class Joint(TypedDict):
    confidence: float
    projection: list[float]
    real: list[float]
    orientation: list[list[float]]


Skeleton = dict[str, Joint | int]


class Poses(TypedDict):
    timestamp: float
    skeletons: list[Skeleton]


def main():
    session_name = "test_session"
    root_dir = Path() / "sessions" / session_name
    poses: list[Poses] = []
    with open(root_dir / "poses.jsonl", "r") as file:
        for line in file:
            poses.append(Poses(json.loads(line)))
    print(f"Loaded {len(poses)} poses")
    keypoints: list[np.ndarray] = []
    timestamps: list[float] = []
    for pose in poses:
        if len(pose["skeletons"]) == 0:
            continue
        skel = pose["skeletons"][0]
        current_keypoints: list[list[float]] = []
        for joint_name in JOINT_NAMES:
            if joint_name not in skel:
                continue
            joint: Joint = skel[joint_name]
            if joint["confidence"] < 0.5:
                continue
            current_keypoints.append(joint["real"])
        if len(current_keypoints) > 0:
            timestamps.append(pose["timestamp"])
            keypoints.append(np.array(current_keypoints))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    (joints,) = ax.plot([], [], [], "ro")

    def init():
        lim_max = np.array([kp.max(axis=0) for kp in keypoints]).max(axis=0)
        lim_min = np.array([kp.min(axis=0) for kp in keypoints]).min(axis=0)
        lim_max += 0.1 * (lim_max - lim_min)
        lim_min -= 0.1 * (lim_max - lim_min)
        ax.set_xlim(lim_min[0], lim_max[0])
        ax.set_ylim(lim_min[2], lim_max[2])
        ax.set_zlim(lim_min[1], lim_max[1])
        return (joints,)

    start = time.time()
    cur_idx = 0

    def update(_frame):
        nonlocal cur_idx
        while (
            time.time() - start > timestamps[cur_idx] - timestamps[0]
            and cur_idx < len(timestamps) - 1
        ):
            cur_idx += 1
        kp = keypoints[cur_idx]
        joints.set_data_3d(kp[:, 0], kp[:, 2], kp[:, 1])
        return (joints,)

    interval = 50
    frames = int((timestamps[-1] - timestamps[0]) * 1000 / interval)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=interval
    )
    plt.show()


if __name__ == "__main__":
    main()
