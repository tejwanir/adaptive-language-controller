import multiprocessing as mp
import time

import numpy as np

from paths import Movement
from ur5 import dt, rtde_c, rtde_r


def admittance_loop(
    action: list[Movement],
    robot_info: "mp.Array[float]",
):
    deadband_trans = 5
    deadband_rot = 0.5

    pose_epsilon = 0.1

    force_scalar = 1

    v_ref = np.zeros(3)
    omega = np.zeros(3)

    movement = action.pop(0)
    initial_pose = movement.start_pose()
    rtde_c.moveL(initial_pose)
    trans_inertia, rot_inertia, driving_force = movement.constants()
    end_pose = movement.end_pose()
    first_time = True

    while True:
        try:
            t_start = rtde_c.initPeriod()
            pose = np.asarray(rtde_r.getActualTCPPose())
            f_tau_raw = np.asarray(rtde_r.getActualTCPForce())
            f_ext = f_tau_raw[:3]
            tau_ext = f_tau_raw[3:]

            robot_info[:] = pose.tolist() + v_ref.tolist() + omega.tolist()

            # Dead banding to catch noise
            if np.linalg.norm(f_ext) < deadband_trans:
                f_ext[:] = 0

            if np.linalg.norm(tau_ext) < deadband_rot:
                tau_ext[:] = 0

            # Calculating Dynamics
            f_virtual, tau_virtual = movement.force_torque(pose, v_ref, omega)

            f_total = (force_scalar * f_virtual) + f_ext
            tau_total = (force_scalar * tau_virtual) + tau_ext

            omega = omega + (dt * tau_total / rot_inertia)
            v_ref = v_ref + dt * (f_total / trans_inertia)

            gen_v = np.concatenate((v_ref, omega))
            rtde_c.speedL(gen_v, 2)

            # parameters that calculate threshold for path completion
            if np.linalg.norm(pose - end_pose) <= pose_epsilon:
                if not movement.paths_left() and not action:
                    rtde_c.speedL(np.zeros(6))
                    break

                if movement.paths_left():
                    movement.update()

                elif action:
                    if first_time:
                        first_time = False
                    movement = action.pop(0)
                    first_time = True
                else:
                    rtde_c.waitPeriod(t_start)
                    continue

                trans_inertia, rot_inertia, driving_force = movement.constants()
                end_pose = movement.end_pose()

            rtde_c.waitPeriod(t_start)
        except BreakAdmittanceControl:
            break
        except KeyboardInterrupt:
            break
    rtde_c.speedL(np.zeros(6))


class BreakAdmittanceControl(RuntimeError):
    pass
