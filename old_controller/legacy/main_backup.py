import copy
import math

from rtde_control import RTDEControlInterface, Path, PathEntry
from rtde_receive import RTDEReceiveInterface
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

ur5_ip = "169.254.9.43"
frequency = 500
dt = 1 / frequency
rtde_c = None
rtde_r = None

rtde_c = RTDEControlInterface(ur5_ip, frequency=frequency)
rtde_r = RTDEReceiveInterface(ur5_ip, frequency=frequency)


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


def impedance_control():
    target = np.array(rtde_r.getActualTCPPose())
    k_pos = 5
    k_tau = 10
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [1, 1, 1, 1, 1, 1]
    limits = [100, 100, 100, 10, 10, 10]
    rtde_c.forceMode(task_frame, selection_vector, [0, 0, 0, 0, 0, 0], 2, limits)

    print("start")
    while True:
        t_start = rtde_c.initPeriod()
        current = np.array(rtde_r.getActualTCPPose())
        pos_error = target[:3] - current[:3]
        rot_initial = R.from_rotvec(target[3:])
        rot_current = R.from_rotvec(current[3:])
        rot_error = (rot_initial * rot_current.inv()).as_rotvec()
        wrench = np.concatenate((k_pos * pos_error, k_tau * rot_error))
        rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)
        rtde_c.waitPeriod(t_start)


def main():
    rtde_c.zeroFtSensor()
    impedance_control()

"""
    initial_pose = rtde_r.getActualTCPPose()
    initial_pose[2] = 0
    rtde_c.moveL(initial_pose)
    initial_pose = teach_mode_pose()
    center = teach_mode_pose()[:3]
    center[2] = 0
    # center = np.array([0, 0, 0])
    rtde_c.moveL(initial_pose)
    angle = np.deg2rad(40)

    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [1, 1, 0, 0, 0, 1]
    limits = [100, 100, 1, 1, 1, 100]
    eps = 0.05
    force = 10
    way_back = False
    
    while True:
        t_start = rtde_c.initPeriod()
        current_pose = np.asarray(rtde_r.getActualTCPPose())
        current_angle = pose_angle(initial_pose, current_pose, center)
        if not way_back:
            if current_angle < angle:
                ideal_pose = rotate_pose(initial_pose, center, [0, 0, current_angle + eps])
                wrench = ideal_pose - np.asarray(current_pose)
                wrench = wrench / np.linalg.norm(wrench[:3]) * force
                wrench[3:] = 0
                rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)
            else:
                way_back = True
        else:
            if current_angle > eps:
                ideal_pose = rotate_pose(initial_pose, center, [0, 0, current_angle - eps])
                wrench = ideal_pose - np.asarray(current_pose)
                wrench = wrench / np.linalg.norm(wrench[:3]) * force
                wrench[3:] = 0
                rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)
            else:
                wrench = [0, 0, 0, 0, 0, 0]
                rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)
                break
        rtde_c.waitPeriod(t_start)

    rtde_c.forceModeStop()
    rtde_c.moveL(initial_pose)
"""

if __name__ == "__main__":
    main()
    rtde_c.disconnect()
    rtde_r.disconnect()
