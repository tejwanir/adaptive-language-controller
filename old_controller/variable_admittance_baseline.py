# import multiprocessing as mp
# from typing import List
#
# import numpy as np
# from ur5 import dt, rtde_r, rtde_c
#
# from paths import Movement
# from language_control import LanguageController
# from force_control import ForceController
#
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def run_admittance_controller(
#         admit_to_sim: "mp.Queue[str]",
#         audio_to_admit: "mp.Queue[str]",
#         admit_to_audio: "mp.Queue[str]",
#         audio_speed: "mp.Value[float]",
#         audio_runtime: "mp.Value[float]",
#         path_speed: "mp.Value[float]",
#         path_runtime: "mp.Array[float,2]",
#         robot_info: "mp.Array[float]",
#         cooperation: "mp.Array[float]",
#         action: List[Movement],
#         lang_control: LanguageController,
#         force_control: ForceController
# ) -> None:
#     # Dead bands are here to ignore noise from force sensor
#
#     audio_speed_data = []
#     path_speed_data = []
#     path_len_data = []
#     audio_len_data = []
#     time_data = []
#     proj_diff_data = []
#     cooperation_data = []
#     path_fidelity_data = []
#     pos_vec_list = []
#     or_vec_list = []
#     f_ext_list = []
#     t_ext_list = []
#     pose_list = []
#     orient_list = []
#
#     raw_audio_speed = 1
#     raw_path_speed = 1
#
#     deadband_trans = 5
#     deadband_rot = 0.5
#
#     pose_epsilon = 0.075
#
#     force_scalar = 1
#
#     v = np.zeros(3)
#     omega = np.zeros(3)
#
#     movement = action.pop(0)
#     initial_pose = movement.start_pose()
#     rtde_c.moveL(initial_pose)
#     trans_inertia, rot_inertia, driving_force = movement.constants()
#     end_pose = movement.end_pose()
#     force_control.set_driving_force(driving_force)
#     first_time = True
#
#     start_time = time.time()
#     print("first movement: ", start_time)
#     while True:
#
#         try:
#
#             t_start = rtde_c.initPeriod()
#
#             pose = np.asarray(rtde_r.getActualTCPPose())
#
#             robot_info[:] = pose.tolist() + v.tolist() + omega.tolist()
#
#             f_tau_raw = np.asarray(rtde_r.getActualTCPForce())
#
#             f_ext = f_tau_raw[:3]
#             tau_ext = f_tau_raw[3:]
#
#             # Dead banding to catch noise
#             if np.linalg.norm(f_ext) < deadband_trans:
#                 f_ext[:] = 0
#
#             if np.linalg.norm(tau_ext) < deadband_rot:
#                 tau_ext[:] = 0
#
#             # Calculating Dynamics
#
#             p = path_speed.value
#
#             f_virtual, tau_virtual = movement.force_torque(pose, v, omega)
#
#             f_total = p*(force_scalar * f_virtual) + f_ext
#             tau_total = p*(force_scalar * tau_virtual) + tau_ext
#
#             omega = omega + (dt * tau_total / rot_inertia)
#             v = v + dt * (f_total / trans_inertia)
#
#             gen_v = np.concatenate((v, omega)) * p
#             rtde_c.speedL(gen_v, 2)
#
#             # AUDIO BLOCK
#
#             path_len = path_runtime[1] - (time.time() - path_runtime[0])
#
#
#             # TODO: DELETE THIS AFTER YOU ARE DONE WITH DATA COLLECTION
#             audio_speed_data.append(audio_speed.value)
#             path_speed_data.append(path_speed.value)
#             path_len_data.append(path_len)
#             audio_len_data.append(audio_runtime.value)
#             path_fidelity_data.append(movement.path_fidelity(pose))
#             proj_diff_data.append(path_len_data[-1] / path_speed_data[-1] - audio_len_data[-1] / audio_speed_data[-1])
#             pos_vec_list.append(pose[:3])
#             or_vec_list.append(pose[3:])
#             f_ext_list.append(f_ext)
#             t_ext_list.append(tau_ext)
#             pose_list.append(pose[:3])
#             orient_list.append(pose[3:])
#
#             if not time_data:
#                 time_data.append(0)
#             else:
#                 time_data.append(time_data[-1] + dt)
#
#
#             cooperation.value = force_control.update_cooperation(f_ext)
#             cooperation.value = max(cooperation.value, 0)
#
#             cooperation_data.append(cooperation.value)
#
#
#             if np.linalg.norm(pose - end_pose) <= pose_epsilon:
#
#                 if not movement.paths_left() and not action:
#                     print("last movement: ", time.time())
#                     admit_to_sim.put("DONE")
#                     rtde_c.speedL(np.zeros(6))
#                     break
#
#                 if movement.paths_left():
#                     admit_to_sim.put("NEW PATH")
#                     movement.update()
#
#
#                 elif action:
#                     if first_time:
#                         print("new movement: ", time.time())
#                         first_time = False
#                     if audio_to_admit.qsize() != 0:
#                         message = audio_to_admit.get()
#
#                         if message == "READY":
#                             admit_to_audio.put("GO")
#                             admit_to_sim.put("NEW MOVEMENT")
#                     movement = action.pop(0)
#                     first_time = True
#                 else:
#                     rtde_c.waitPeriod(t_start)
#                     continue
#
#                 trans_inertia, rot_inertia, driving_force = movement.constants()
#                 end_pose = movement.end_pose()
#                 force_control.set_driving_force(driving_force)
#
#             rtde_c.waitPeriod(t_start)
#         except BreakAdmittanceControl:
#             break
#         except KeyboardInterrupt:
#             break
#     rtde_c.speedL(np.zeros(6))
#     end_time = time.time()
#
#     total_time = end_time - start_time
#     print("total time: ", total_time)
#
#     time_to_completion = [(total_time - x) for x in time_data]
#     path_speed_data = [10 * x for x in path_speed_data]
#     audio_speed_data = [10 * x for x in audio_speed_data]
#     cooperation_data = [10 * x for x in cooperation_data]
#     path_fidelity_data = [10 * x for x in path_fidelity_data]
#     #
#     plt.plot(time_data, audio_speed_data, label='audio speed')
#     plt.plot(time_data, path_speed_data, label='path speed')
#     #plt.plot(time_data, audio_len_data, label='audio len')
#     #plt.plot(time_data, path_len_data, label='path len')
#     # plt.plot(time_data, time_to_completion, label='time to completion')
#     #plt.plot(time_data, proj_diff_data, label='proj diff')
#     plt.plot(time_data, cooperation_data, label='cooperation')
#     #plt.plot(time_data, path_fidelity_data, label='path_fidelity')
# #    Adding labels and a legend
#
#     print("pose_list:", pose_list[-1])
#     print("orient_list:",orient_list[-1])
#     print("f_ext_list:",f_ext_list[-1])
#     print("t_ext_list:",t_ext_list[-1])
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.legend()
#     plt.show()
#
#     columns = {
#         "time": time_data,
#         "audio_speed": audio_speed_data,
#         "path_speed": path_speed_data,
#         "path_len": path_len_data,
#         "audio_len": audio_len_data,
#         "proj_diff": proj_diff_data,
#         "coop": cooperation_data,
#         "path_fidelity": path_fidelity_data,
#     }
#
#     def add_vec_list(name, vl):
#         columns[name + "_1"] = [x[0] for x in vl]
#         columns[name + "_2"] = [x[1] for x in vl]
#         columns[name + "_3"] = [x[2] for x in vl]
#
#     add_vec_list("pose", pose_list)
#     add_vec_list("orient", orient_list)
#     add_vec_list("f_ext", f_ext_list)
#     add_vec_list("t_ext", t_ext_list)
#     import pandas as pd
#     df = pd.DataFrame(columns)
#     import datetime
#     csv_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
#     df.to_csv(csv_file)
#
# class BreakAdmittanceControl(RuntimeError):
#     pass
