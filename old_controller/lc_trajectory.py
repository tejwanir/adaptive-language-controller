import multiprocessing as mp

from typing import List

from ur5 import gripper, initialize_ur5, rtde_c

from lc_admittance import run_admittance_controller
from sound import audio_processor
from path_sim import update_path_runtime

from paths import Movement
from resistance import ResistanceTracker
from language_control import LanguageController
from phrasing_digraph import WordGraph

from global_constants import normal_turn, slow_turn, normal_walk, slow_walk, test_lang_controller, test_force_controller
from global_constants import test_head
from global_constants import baseline_head
import time


def run_trajectory(action: List[Movement], speach: List[WordGraph],
                   lang_controller: LanguageController, resistance_tracker: ResistanceTracker) -> None:
    """

    @param action: an action ( formatted as a list of movements) for the robot to perform
    @param speach: A speach (formatted as a list of phrasing digraphs) for the robot to say along with the action
    @param lang_controller: The language controller used to regulate how the physical and audio pace
    @return:
    """
    initialize_ur5()
    print("Pre-gripper moved")
    gripper.move_and_wait_for_pos(105, 255, 1)
    print("Gripper moved")
    rtde_c.zeroFtSensor()
    print("Force sensor zero'd")

    movement = action[0]
    path_runtime = mp.Array("d", [time.time(), movement.total_runtime()])
    path_speed = mp.Value("d", 1)
    cooperation = mp.Value("d", 0.2)

    word_graph = speach[0]
    audio_runtime = mp.Value("d", word_graph.exp_rem_len(path_runtime[1], 1))
    audio_speed = mp.Value("d", 1)

    init_pose = movement.start_pose()
    init_info = init_pose.tolist() + [0] * 9
    robot_info = mp.Array("d", init_info)

    # MESSAGE QUEUES
    admit_to_audio = mp.Queue()
    audio_to_admit = mp.Queue()
    admit_to_sim = mp.Queue()

    rtde_c.moveL(init_pose)

    process_1 = mp.Process(target=audio_processor, args=[audio_speed, audio_runtime, speach, admit_to_audio,
                                                         audio_to_admit, path_runtime, cooperation])

    process_2 = mp.Process(target=update_path_runtime, args=[action, admit_to_sim, robot_info, path_runtime])

    process_1.start()
    process_2.start()

    run_admittance_controller(
        admit_to_sim=admit_to_sim,
        admit_to_audio=admit_to_audio,
        audio_to_admit=audio_to_admit,
        audio_pace=audio_speed,
        audio_runtime=audio_runtime,

        physical_pace=path_speed,
        path_runtime=path_runtime,
        robot_info=robot_info,
        resistance=cooperation,
        action=action,
        lang_control=lang_controller,
        force_control=resistance_tracker
    )

if __name__ == "__main__":

    mov_1 = Movement([normal_turn])
    mov_2 = Movement([normal_walk])
    mov_3 = Movement([normal_turn])
    mov_4 = Movement([normal_walk])

    slow_mov_1 = Movement([slow_turn])
    slow_mov_2 = Movement([slow_walk])
    slow_mov_3 = Movement([slow_turn])
    slow_mov_4 = Movement([slow_walk])


    demo_mov = Movement([slow_turn, slow_walk, slow_turn])

    demo_action = [demo_mov]

    lc = [test_head]
    lc_noap = [baseline_head]

    run_trajectory(demo_action, lc, test_lang_controller, test_force_controller)
