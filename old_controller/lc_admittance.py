import multiprocessing as mp
from typing import List

import numpy as np
from ur5 import dt, rtde_r, rtde_c

from paths import Movement
from language_control import LanguageController
from resistance import ResistanceTracker

import time
import matplotlib.pyplot as plt


def run_admittance_controller(
        admit_to_sim: "mp.Queue[str]",
        audio_to_admit: "mp.Queue[str]",
        admit_to_audio: "mp.Queue[str]",
        audio_pace: "mp.Value[float]",
        audio_runtime: "mp.Value[float]",
        physical_pace: "mp.Value[float]",
        path_runtime: "mp.Array[float,2]",
        robot_info: "mp.Array[float]",
        resistance: "mp.Array[float]",
        action: List[Movement],
        lang_control: LanguageController,
        force_control: ResistanceTracker
) -> None:
    # Dead bands are here to ignore noise from force sensor
    # P = the physical pace and v_ref = the reference velocity

    audio_pace_data = []
    physical_pace_data = []
    path_len_data = []
    audio_len_data = []
    time_data = []
    proj_diff_data = []
    resistance_data = []
    path_fidelity_data = []
    pos_vec_list = []
    or_vec_list = []
    f_ext_list = []
    t_ext_list = []
    pose_list = []
    orient_list = []

    raw_audio_speed = 1
    raw_path_speed = 1

    deadband_trans = 5
    deadband_rot = 0.5

    pose_epsilon = 0.1

    v_ref = np.zeros(3)
    omega = np.zeros(3)

    movement = action.pop(0)
    initial_pose = movement.start_pose()
    rtde_c.moveL(initial_pose)
    trans_inertia, rot_inertia, driving_force = movement.constants()
    end_pose = movement.end_pose()
    first_time = True

    start_time = time.time()
    print("first movement: ", start_time)
    while True:

        try:

            t_start = rtde_c.initPeriod()

            pose = np.asarray(rtde_r.getActualTCPPose())

            f_tau_raw = np.asarray(rtde_r.getActualTCPForce())

            f_ext = f_tau_raw[:3]
            tau_ext = f_tau_raw[3:]

            # Dead banding to catch noise
            if np.linalg.norm(f_ext) < deadband_trans:
                f_ext[:] = 0

            if np.linalg.norm(tau_ext) < deadband_rot:
                tau_ext[:] = 0

            robot_info[:] = pose.tolist() + v_ref.tolist() + omega.tolist() + f_ext.tolist()

            # Calculating Dynamics

            p = physical_pace.value

            f_virtual, tau_virtual = movement.force_torque(pose, v_ref, omega)

            f_total = p*(f_virtual) + f_ext
            tau_total = p*(tau_virtual) + tau_ext

            omega = omega + (dt * tau_total / rot_inertia)
            v_ref = v_ref + dt * (f_total / trans_inertia)

            gen_v = np.concatenate((v_ref, omega)) * p
            rtde_c.speedL(gen_v, 2)

            # AUDIO BLOCK

            path_len = path_runtime[1] - (time.time() - path_runtime[0])

            raw_audio_speed, raw_path_speed, audio_pace.value, physical_pace.value = lang_control.harmonic_control(
                audio_runtime.value, raw_audio_speed, path_len, raw_path_speed, resistance.value)

            raw_audio_speed = audio_pace.value
            raw_path_speed = physical_pace.value



            # This portion of the code does not affect the movement of the robot. It is used purely for data collection
            audio_pace_data.append(audio_pace.value)
            physical_pace_data.append(physical_pace.value)
            path_len_data.append(path_len)
            audio_len_data.append(audio_runtime.value)
            path_fidelity_data.append(movement.path_fidelity(pose))
            proj_diff_data.append(path_len_data[-1] / physical_pace_data[-1] - audio_len_data[-1] / audio_pace_data[-1])
            pos_vec_list.append(pose[:3])
            or_vec_list.append(pose[3:])
            f_ext_list.append(f_ext)
            t_ext_list.append(tau_ext)
            pose_list.append(pose[:3])
            orient_list.append(pose[3:])

            if not time_data:
                time_data.append(0)
            else:
                time_data.append(time_data[-1] + dt)

            resistance.value = force_control.update_resistance(f_ext)
            resistance.value = max(resistance.value, 0)

            resistance_data.append(resistance.value)

            if np.linalg.norm(pose - end_pose) <= pose_epsilon:

                if not movement.paths_left() and not action:
                    print("last movement: ", time.time())
                    admit_to_sim.put("DONE")
                    rtde_c.speedL(np.zeros(6))
                    break

                if movement.paths_left():
                    admit_to_sim.put("NEW PATH")
                    movement.update()


                elif action:
                    if first_time:
                        print("new movement: ", time.time())
                        first_time = False
                    if audio_to_admit.qsize() != 0:
                        message = audio_to_admit.get()

                        if message == "READY":
                            admit_to_audio.put("GO")
                            admit_to_sim.put("NEW MOVEMENT")
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

    # Some data is rescaled here so it can all be seen on the same plot.
    # The rescaled data is used only for visualization purposes. All analysis, was done on data that was scaled back to its original dimensions

    physical_pace_data = [10 * x for x in physical_pace_data]
    audio_pace_data = [10 * x for x in audio_pace_data]
    resistance_data = [10 * x for x in resistance_data]
    path_fidelity_data = [10 * x for x in path_fidelity_data]
    #

    actual_speed = []
    for i, pose in enumerate(pose_list):
        if i == 0:
            actual_speed.append(0)
        else:
            actual_speed.append(np.linalg.norm(pose - pose_list[i - 1]) / (time_data[i] - time_data[i - 1]))

    print("pose_list:", pose_list[-1])
    print("orient_list:",orient_list[-1])
    print("f_ext_list:",f_ext_list[-1])
    print("t_ext_list:",t_ext_list[-1])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

    columns = {
        "time": time_data,
        "audio_speed": audio_pace_data,
        "path_speed": physical_pace_data,
        "path_len": path_len_data,
        "audio_len": audio_len_data,
        "proj_diff": proj_diff_data,
        "coop": resistance_data,
        "path_fidelity": path_fidelity_data,
    }

    def add_vec_list(name, vl):
        columns[name + "_1"] = [x[0] for x in vl]
        columns[name + "_2"] = [x[1] for x in vl]
        columns[name + "_3"] = [x[2] for x in vl]

    add_vec_list("pose", pose_list)
    add_vec_list("orient", orient_list)
    add_vec_list("f_ext", f_ext_list)
    add_vec_list("t_ext", t_ext_list)
    import pandas as pd
    df = pd.DataFrame(columns)
    import datetime
    csv_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    df.to_csv(csv_file)


class BreakAdmittanceControl(RuntimeError):
    pass
