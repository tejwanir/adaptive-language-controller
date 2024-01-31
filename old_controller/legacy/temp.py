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
far_pose = np.array(
    [
        -0.6836371243195373,
        -0.7249986525872217,
        0.5234799041622413,
        -0.3968537654978812,
        -1.4239596450982468,
        2.312984780018482,
    ]
)
initial_pose = np.array(
    z_align_pose(
        [
            -0.350,
            -0.456,
            -0.030,
            0.16709463563678736,
            -1.9706016156632593,
            0.22494408451669762,
        ]
    )
)
print(rtde_r.getActualTCPPose())
new_pose = np.copy(initial_pose)
new_pose[2] += 0.05
gripper.move(2, 255, 1)
rtde_c.moveL(new_pose)
rtde_c.moveL(initial_pose)
gripper.move_and_wait_for_pos(108, 255, 1)
rtde_c.moveL(far_pose, 1)
time.sleep(1)
rtde_c.moveL(new_pose, 1)
gripper.move(2, 255, 1)
