import atexit
import time

import numpy as np

from rtde_control import RTDEControlInterface, Path, PathEntry
from rtde_receive import RTDEReceiveInterface
from gripper import RobotiqGripper

from utils import rotate_pose, z_align_pose, disconnect_at_exit

ur5_ip = "169.254.9.43"
frequency = 500
rtde_c = RTDEControlInterface(ur5_ip, frequency=frequency)
rtde_r = RTDEReceiveInterface(ur5_ip, frequency=frequency)
disconnect_at_exit(rtde_c, rtde_r)
gripper = RobotiqGripper()
gripper.connect(ur5_ip, 63352)
print(rtde_r.getActualTCPPose())

far_pose = np.array(
[-0.7076532342716697, -0.7497372490329921, 0.5521916533507686, -0.6366938844274882, -1.560667980593342, 2.0838697074341743]
)
initial_pose = np.array(
    [
        -0.350,
        -0.456,
        0.010,
        0.16709463563678736,
        -1.9706016156632593,
        0.22494408451669762,
    ]
)
aligned_pose = z_align_pose(initial_pose)
rtde_c.zeroFtSensor()
rtde_c.moveL(aligned_pose, 1)
gripper.move_and_wait_for_pos(104, 1, 0)
rtde_c.moveL(initial_pose, 0.5)
paths = []
for theta in np.linspace(0, 40, 10):
    rotated_pose = rotate_pose(initial_pose, [-0.400, -0.206, 0], [0, 0, -np.deg2rad(theta)])
    paths.append(np.concatenate((rotated_pose, [0.05, 1.2, 0.01])))
input()
rtde_c.moveL(paths)
paths = []
for theta in np.linspace(40, 0, 10):
    rotated_pose = rotate_pose(initial_pose, [-0.400, -0.206, 0], [0, 0, -np.deg2rad(theta)])
    paths.append(np.concatenate((rotated_pose, [0.05, 1.2, 0.01])))
rtde_c.moveL(paths)
exit(0)
# rtde_c.moveL(far_pose)


def disconnect_session():
    global rtde_c, rtde_r
    rtde_c.disconnect()
    rtde_r.disconnect()


atexit.register(disconnect_session)

if False:
    while True:
        force = rtde_r.getActualTCPForce()[:3]
        deadband = 3
        print(np.linalg.norm(force))
        time.sleep(0.05)
