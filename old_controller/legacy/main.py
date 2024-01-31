import copy
import math
import atexit
from typing import Optional

from rtde_control import RTDEControlInterface, Path, PathEntry, Flags
from rtde_receive import RTDEReceiveInterface
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

ur5_ip = "169.254.9.43"
frequency = 500
dt = 1 / frequency
rtde_c: Optional[RTDEControlInterface] = None
rtde_r: Optional[RTDEReceiveInterface] = None


def rotate_pose(
    pose: npt.ArrayLike, center: npt.ArrayLike, rotvec: npt.ArrayLike
) -> npt.ArrayLike:
    r = R.from_rotvec(rotvec)
    pose = np.asarray(pose)
    center = np.asarray(center)
    return np.hstack(
        (r.apply(pose[:3] - center) + center, (r * R.from_rotvec(pose[3:])).as_rotvec())
    )


def pose_angle(a: npt.ArrayLike, b: npt.ArrayLike, center: npt.ArrayLike) -> float:
    a, b = np.asarray(a), np.asarray(b)
    a = a[:3] - np.asarray(center)
    b = b[:3] - np.asarray(center)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.asin(np.linalg.norm(np.cross(a, b)))


def get_unit_wrench(pose: npt.ArrayLike, center: npt.ArrayLike, axis: npt.ArrayLike):
    pose = np.asarray(pose)[:3]
    center = np.asarray(center)[:3]
    axis = np.asarray(axis)
    c = np.cross(axis, pose - center)
    return np.hstack((c / np.linalg.norm(c), np.array([0, 0, 0])))


def teach_mode_pose() -> np.ndarray:
    rtde_c.teachMode()
    input("enter to end teach mode")
    rtde_c.endTeachMode()
    pose = np.array(rtde_r.getActualTCPPose())
    pose[[2, 5]] = 0
    pose[3:5] = pose[3:5] / np.linalg.norm(pose[3:5]) * math.pi
    return pose


def impedance_control(callback):
    rtde_c.zeroFtSensor()
    k_pos, k_tau = 5, 10
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [1, 1, 1, 1, 1, 1]
    limits = [100, 100, 100, 10, 10, 10]
    rtde_c.forceMode(task_frame, selection_vector, [0, 0, 0, 0, 0, 0], 2, limits)

    print("start")
    while True:
        t_start = rtde_c.initPeriod()
        current = np.array(rtde_r.getActualTCPPose())
        target = callback(current)
        pos_error = target[:3] - current[:3]
        rot_initial = R.from_rotvec(target[3:])
        rot_current = R.from_rotvec(current[3:])
        rot_error = (rot_initial * rot_current.inv()).as_rotvec()
        wrench = np.concatenate((k_pos * pos_error, k_tau * rot_error))
        rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)
        rtde_c.waitPeriod(t_start)


def main():
    global rtde_c, rtde_r
    # noinspection PyArgumentList
    rtde_c = RTDEControlInterface(ur5_ip, frequency=frequency)
    # noinspection PyArgumentList
    rtde_r = RTDEReceiveInterface(ur5_ip, frequency=frequency)

    def disconnect_session():
        rtde_c.disconnect()
        rtde_r.disconnect()

    atexit.register(disconnect_session)
    the_target = np.array(rtde_r.getActualTCPPose())
    initial_pose = rtde_r.getActualTCPPose()
    center = np.array([0, 0, 0])
    max_angle = np.deg2rad(30)
    eps = 0.05

    def callback(current_pose: np.ndarray) -> npt.ArrayLike:
        current_angle = pose_angle(initial_pose, current_pose, center)
        if current_angle < max_angle:
            ideal_pose = rotate_pose(initial_pose, center, [0, 0, current_angle + eps])
            return ideal_pose
        return rotate_pose(initial_pose, center, [0, 0, max_angle])

    impedance_control(callback)


if __name__ == "__main__":
    main()
