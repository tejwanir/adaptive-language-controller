# A shim around UR-RTDE

import atexit
from typing import Optional

from gripper import RobotiqGripper
from third_party.ur_rtde.rtde_control import RTDEControlInterface
from third_party.ur_rtde.rtde_receive import RTDEReceiveInterface

ur5_ip = "169.254.9.43"
frequency = 500
dt = 1 / frequency
rtde_c: Optional[RTDEControlInterface] = None
rtde_r: Optional[RTDEReceiveInterface] = None
gripper = RobotiqGripper()


def disconnect_at_exit(
    rtde_c: RTDEControlInterface, rtde_r: RTDEReceiveInterface
) -> None:
    def disconnect():
        rtde_c.disconnect()
        rtde_r.disconnect()

    atexit.register(disconnect)


# TODO: DONT RUN IN SUBPROCESS!
def initialize_ur5():
    global rtde_c, rtde_r
    if rtde_c is None or rtde_r is None:
        rtde_c = RTDEControlInterface(ur5_ip, frequency=frequency)
        rtde_r = RTDEReceiveInterface(ur5_ip, frequency=frequency)
        gripper.connect(ur5_ip, 63352)
        disconnect_at_exit(rtde_c, rtde_r)


initialize_ur5()
