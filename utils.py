import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def sigmoid(x):
    return 2 / (1 + math.exp(-x / 2)) - 1


def rotate_pose(
    pose: npt.ArrayLike, center: npt.ArrayLike, rotvec: npt.ArrayLike
) -> np.ndarray:
    """
    Rotates a pose around center.


    :param pose: The pose (a 6-array).
    :param center: The center of rotation.
    :param rotvec: The rotation, in rotation vector format.
    :return: The rotated pose.
    """
    r = R.from_rotvec(rotvec)
    pose = np.asarray(pose)
    center = np.asarray(center)
    return np.hstack(
        (r.apply(pose[:3] - center) + center, (r * R.from_rotvec(pose[3:])).as_rotvec())
    )


def z_align_pose(pose: npt.ArrayLike) -> npt.ArrayLike:
    """
    Aligns a pose to Z axis. Similar to the "Align" button on the UR5 controller.


    :param pose: The pose (a 6-array).
    :return: The aligned pose.
    """
    pose = np.array(pose)
    pose[3:] = 0
    pose[4] = np.pi
    return pose


def angle_between(a: npt.ArrayLike, b: npt.ArrayLike, center: npt.ArrayLike) -> float:
    """
    Computes the angle between two 3-D points around the given center.


    :param a: One point.
    :param b: Another point.
    :param center: The center.
    :return: The angle between (a-center) and (b-center). The sign is determined as follows:
             If looking from above (i.e., looking at the origin from positive z-axis), a is in ccw direction of b, then
             the returned angle is positive; false otherwise.
    """
    a, b = np.asarray(a), np.asarray(b)
    a = a - np.asarray(center)
    b = b - np.asarray(center)
    c = np.cross(a / np.linalg.norm(a), b / np.linalg.norm(b))
    return math.asin(np.linalg.norm(c)) * np.sign(-c[2])


def sim_speedL(
    pose: np.ndarray,
    cur_v: np.ndarray,
    target_v: np.ndarray,
    accel: float,
    sim_dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the speedL command on the UR5
    Accelerates linearly from the current to target velocity adn then remains at target velocity until
    the end of the time step(whose length is given by sim_dt)
    Returns the updated pose of the (simulated) UR5
    """

    # Translational component

    trans_v = cur_v[:3]
    trans_pose = pose[:3]
    target_trans_v = target_v[:3]

    trans_diff_vec = target_trans_v - trans_v
    trans_a_time = np.linalg.norm(trans_diff_vec) / accel

    if np.linalg.norm(trans_diff_vec) != 0:
        trans_a_vec = trans_diff_vec / np.linalg.norm(trans_diff_vec)
    else:
        trans_a_vec = np.zeros(3)

    trans_a_time = min(sim_dt, trans_a_time)
    trans_del1 = trans_v * trans_a_time + accel / 2 * trans_a_vec * (trans_a_time * 2)
    trans_del2 = target_trans_v * (sim_dt - trans_a_time)
    new_trans_pose = trans_pose + trans_del1 + trans_del2
    new_v = trans_v + trans_a_time * trans_a_vec

    # Rotational component
    rot_v = cur_v[3:]
    target_rot_v = target_v[3:]
    rot_pose = pose[3:]

    rot_diff_vec = target_rot_v - rot_v
    rot_accel = 100
    rot_a_time = np.linalg.norm(rot_diff_vec) / rot_accel

    if np.linalg.norm(rot_diff_vec) != 0:
        rot_a_vec = rot_diff_vec / np.linalg.norm(rot_diff_vec)
    else:
        rot_a_vec = np.zeros(3)

    rot_a_time = min(sim_dt, rot_a_time)
    rot_del1 = rot_v * rot_a_time + rot_accel / 2 * rot_a_vec * (rot_a_time * 2)
    rot_del2 = target_rot_v * (sim_dt - rot_a_time)
    new_rot_pose = rot_pose + rot_del1 + rot_del2
    new_pose = np.concatenate((new_trans_pose, new_rot_pose))
    new_omega = rot_v + rot_a_time * rot_a_vec

    return new_pose, np.concatenate([new_v, new_omega])
