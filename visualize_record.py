import json
from pathlib import Path
from typing import TypedDict, List, Dict, Union
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Assuming JOINT_NAMES is a list of joint names
from record import JOINT_NAMES, JOINT_CONNECTIONS

class Joint(TypedDict):
    confidence: float
    projection: List[float]
    real: List[float]
    orientation: List[List[float]]

Skeleton = Dict[str, Union[Joint, int]]

class Poses(TypedDict):
    timestamp: float
    skeletons: List[Skeleton]

def main():

    session_name = "lab_session_7" # CHANGE SESSION NAME TO VISUALIZE

    root_dir = Path() / "sessions" / session_name
    poses: List[Poses] = []
    with open(root_dir / "cut_poses.jsonl", "r") as file:
        for line in file:
            poses.append(Poses(json.loads(line)))
    print(f"Loaded {len(poses)} poses")

    # Store keypoints for each skeleton in each pose
    all_keypoints: List[List[np.ndarray]] = []
    timestamps: List[float] = []

    for pose in poses:
        if len(pose["skeletons"]) == 0:
            continue
        pose_keypoints: List[np.ndarray] = []
        for skel in pose["skeletons"]:
            current_keypoints: List[List[float]] = []
            for joint_name in JOINT_NAMES:
                if joint_name not in skel:
                    continue
                joint: Joint = skel[joint_name]
                if joint["confidence"] <= 0.5:
                    continue
                current_keypoints.append(joint["real"])
            if len(current_keypoints) > 0:
                pose_keypoints.append(np.array(current_keypoints))
        if pose_keypoints:
            timestamps.append(pose["timestamp"])
            all_keypoints.append(pose_keypoints)

    fig = plt.figure(figsize=(20, 12))  # Adjust the figure size as needed
    ax = fig.add_subplot(111, projection="3d")

    # Initialize red dots
    dot_lines = [ax.plot([], [], [], "ro")[0] for _ in range(len(all_keypoints[0]))]

    # Initialize green lines
    connection_lines = [[ax.plot([], [], [], 'g')[0] for _ in JOINT_CONNECTIONS] for _ in range(len(all_keypoints[0]))]

    # New function to get joint index by name
    def get_joint_index(name):
        return JOINT_NAMES.index(name)

    def init():
        lim_max = np.array([kp.max(axis=0) for pose_kps in all_keypoints for kp in pose_kps]).max(axis=0)
        lim_min = np.array([kp.min(axis=0) for pose_kps in all_keypoints for kp in pose_kps]).min(axis=0)
        
        zoom_factor = 0.5  # Adjust this value to control the zoom level
        lim_max += 0.5 * zoom_factor * (lim_max - lim_min)
        lim_min -= 0.5 * zoom_factor * (lim_max - lim_min)
        
        ax.set_xlim(lim_min[0], lim_max[0])
        ax.set_ylim(lim_min[2], lim_max[2])
        ax.set_zlim(lim_min[1], lim_max[1])
        
        return dot_lines + [line for skeleton_lines in connection_lines for line in skeleton_lines]

    start = time.time()
    cur_idx = 0

    # Modify the update function to update connection lines for each skeleton
    def update(_frame):
        nonlocal cur_idx
        while (
            time.time() - start > timestamps[cur_idx] - timestamps[0]
            and cur_idx < len(timestamps) - 1
        ):
            cur_idx += 1

        for line, kp in zip(dot_lines, all_keypoints[cur_idx]):
            line.set_data_3d(kp[:, 0], kp[:, 2], kp[:, 1])

        for skeleton_index, skeleton in enumerate(all_keypoints[cur_idx]):
            for conn_line, (joint1, joint2) in zip(connection_lines[skeleton_index], JOINT_CONNECTIONS):
                try:
                    idx1 = get_joint_index(joint1)
                    idx2 = get_joint_index(joint2)
                    conn_line.set_data_3d(
                        [skeleton[idx1, 0], skeleton[idx2, 0]],
                        [skeleton[idx1, 2], skeleton[idx2, 2]],
                        [skeleton[idx1, 1], skeleton[idx2, 1]]
                    )
                except IndexError:
                    # This happens if one of the joints is not found in the current skeleton
                    conn_line.set_data_3d([], [], [])

        return dot_lines + [line for skeleton_lines in connection_lines for line in skeleton_lines]

    interval = 50
    frames = int((timestamps[-1] - timestamps[0]) * 1000 / interval)

    # Set up different camera perspectives
    perspectives = [
        {"azim": -90, "elev": 0, "filename": str(root_dir / "front_perspective.mp4")},
        {"azim": -90, "elev": 90, "filename": str(root_dir / "top_perspective.mp4")},
    ]

    print("Recording starting")

    for perspective in perspectives:
        print("Recording next perspective")
        ax.view_init(azim=perspective["azim"], elev=perspective["elev"])
        ani = FuncAnimation(
            fig, update, frames=frames, init_func=init, blit=True, interval=interval
        )
        ani.save(perspective["filename"], writer="ffmpeg", fps=30)

if __name__ == "__main__":
    main()
