import json
from pathlib import Path
from typing import TypedDict, List, Dict, Union
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Assuming JOINT_NAMES is a list of joint names and JOINT_CONNECTIONS is a list of joint connections
from record import JOINT_NAMES, JOINT_CONNECTIONS

class Joint(TypedDict):
    confidence: float
    projection: List[float]
    pos3D: List[float]
    orientation: List[List[float]]

Skeleton = Dict[str, Union[Joint, int]]

class Poses(TypedDict):
    timestamp: float
    skeletons: List[Skeleton]

def main(session_names: List[str]):
    for session_name in session_names:
        root_dir = Path() / "sessions" / session_name
        poses: List[Poses] = []
        with open(root_dir / "cut_poses.jsonl", "r") as file:
            for line in file:
                poses.append(json.loads(line))
        print(f"Loaded {len(poses)} poses from session {session_name}")

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
                    current_keypoints.append(joint["pos3D"])  # Assuming 'pos3D' is the 3D position key
                if len(current_keypoints) > 0:
                    pose_keypoints.append(np.array(current_keypoints))
            if pose_keypoints:
                timestamps.append(pose["timestamp"])
                all_keypoints.append(pose_keypoints)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        lines = [ax.plot([], [], [], "ro")[0] for _ in range(len(all_keypoints[0]))]

        lim_max = np.array([kp.max(axis=0) for pose_kps in all_keypoints for kp in pose_kps]).max(axis=0)
        lim_min = np.array([kp.min(axis=0) for pose_kps in all_keypoints for kp in pose_kps]).min(axis=0)
        lim_max += 0.1 * (lim_max - lim_min)
        lim_min -= 0.1 * (lim_max - lim_min)

        def init():
            ax.set_xlim(lim_min[0], lim_max[0])
            ax.set_ylim(lim_min[2], lim_max[2])
            ax.set_zlim(lim_min[1], lim_max[1])
            ax.view_init(elev=0, azim=90)  # Set elevation to 0 and azimuth to 90 for front view
            return lines

        start = time.time()
        cur_idx = 0

        def get_joint_index(name):
            return JOINT_NAMES.index(name)

        dot_lines = [ax.plot([], [], [], "ro")[0] for _ in range(len(all_keypoints[0]))]
        connection_lines = [[ax.plot([], [], [], 'g')[0] for _ in JOINT_CONNECTIONS] for _ in range(len(all_keypoints[0]))]

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
                        if joint1 == "WristRight" or joint2 == "WristRight":
                            conn_line.set_color('violet')
                        elif joint1 == "WristLeft" or joint2 == "WristLeft":
                            conn_line.set_color('blue')
                        else:
                            conn_line.set_color('green')
                            conn_line.set_alpha(0.5)
                        conn_line.set_data_3d(
                            [skeleton[idx1, 0], skeleton[idx2, 0]],
                            [skeleton[idx1, 2], skeleton[idx2, 2]],
                            [skeleton[idx1, 1], skeleton[idx2, 1]]
                        )
                    except IndexError:
                        conn_line.set_data_3d([], [], [])

            return dot_lines + [line for skeleton_lines in connection_lines for line in skeleton_lines]

        interval = 33.33
        frames = int((timestamps[-1] - timestamps[0]) * 1000 / interval)
        front_ani = FuncAnimation(
            fig, update, frames=frames, init_func=init, blit=True, interval=interval
        )

        top_view_fig = plt.figure()
        top_view_ax = top_view_fig.add_subplot(111, projection="3d")
        top_view_lines = [top_view_ax.plot([], [], [], "ro")[0] for _ in range(len(all_keypoints[0]))]
        top_view_connection_lines = [[top_view_ax.plot([], [], [], 'g')[0] for _ in JOINT_CONNECTIONS] for _ in range(len(all_keypoints[0]))]

        def top_view_init():
            top_view_ax.set_xlim(lim_min[0], lim_max[0])
            top_view_ax.set_ylim(lim_min[1], lim_max[1])
            top_view_ax.set_zlim(lim_min[2], lim_max[2])
            top_view_ax.view_init(elev=90, azim=90)  # Set elevation to 90 for top view
            return top_view_lines + [line for skeleton_lines in top_view_connection_lines for line in skeleton_lines]

        def top_view_update(_frame):
            nonlocal cur_idx
            while (
                time.time() - start > timestamps[cur_idx] - timestamps[0]
                and cur_idx < len(timestamps) - 1
            ):
                cur_idx += 1
            
            for line, kp in zip(top_view_lines, all_keypoints[cur_idx]):
                line.set_data_3d(kp[:, 0], kp[:, 1], kp[:, 2])

            for skeleton_index, skeleton in enumerate(all_keypoints[cur_idx]):
                for conn_line, (joint1, joint2) in zip(top_view_connection_lines[skeleton_index], JOINT_CONNECTIONS):
                    try:
                        idx1 = get_joint_index(joint1)
                        idx2 = get_joint_index(joint2)
                        if joint1 == "WristRight" or joint2 == "WristRight":
                            conn_line.set_color('violet')
                        elif joint1 == "WristLeft" or joint2 == "WristLeft":
                            conn_line.set_color('blue')
                        else:
                            conn_line.set_color('green')
                            conn_line.set_alpha(0.5)
                        conn_line.set_data_3d(
                            [skeleton[idx1, 0], skeleton[idx2, 0]],
                            [skeleton[idx1, 1], skeleton[idx2, 1]],
                            [skeleton[idx1, 2], skeleton[idx2, 2]]
                        )
                    except IndexError:
                        conn_line.set_data_3d([], [], [])

            return top_view_lines + [line for skeleton_lines in top_view_connection_lines for line in skeleton_lines]

        top_view_interval = 33.33
        top_view_frames = int((timestamps[-1] - timestamps[0]) * 1000 / top_view_interval)
        top_view_ani = FuncAnimation(
            top_view_fig, top_view_update, frames=top_view_frames, init_func=top_view_init, blit=True, interval=top_view_interval
        )

        front_ani.save(root_dir / "body_tracking_front.mp4")
        top_view_ani.save(root_dir / "body_tracking_top.mp4")

if __name__ == "__main__":
    session_names = ["lightbuzz_table_1", "lightbuzz_table_2", "lightbuzz_table_3", "lightbuzz_table_4", "lightbuzz_table_5", "lightbuzz_table_6"]  # Add session names here
    main(session_names)
