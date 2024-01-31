from typing import Optional
import multiprocessing as mp

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from gripper import RobotiqGripper
from utils import disconnect_at_exit

import os

ur5_ip = "169.254.9.43"
frequency = 500
dt = 1 / frequency
rtde_c: Optional[RTDEControlInterface] = None
rtde_r: Optional[RTDEReceiveInterface] = None
gripper = RobotiqGripper()


# TODO: DONT RUN IN SUBPROCESS!
def initialize_ur5():
    global rtde_c, rtde_r
    print(mp.parent_process(), os.getpid())
    if rtde_c is None or rtde_r is None:
        print("initialized")
        rtde_c = RTDEControlInterface(ur5_ip, frequency=frequency)
        rtde_r = RTDEReceiveInterface(ur5_ip, frequency=frequency)
        gripper.connect(ur5_ip, 63352)
        disconnect_at_exit(rtde_c, rtde_r)


initialize_ur5()
