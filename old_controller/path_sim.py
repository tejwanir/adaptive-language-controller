import multiprocessing as mp
from typing import List

import time
import numpy as np
from paths import Movement


def update_path_runtime(action: List[Movement], message_queue: "mp.Queue[str]", robot_info: "mp.Array[np.ndarray, 2]",
                        path_runtime: "mp.Array[float, float]") -> None:
    """
    @param paths: the list of paths that will be followed
    @param message_queue: A multiprocessing queue used to coordinate with admittance control
    @param robot_info: the TCP position/orientation and linear/angular velocity at any given moment
    @param path_runtime: An array used to communicate the expected runtime
    This function continually updates the estimated runtime in order to facilitate coordination between robot/ audio
    speed
    """

    movement = action.pop(0)

    while True:

        if message_queue.qsize() != 0:
            message = message_queue.get()
            if message == "DONE":
                break
            if message == "NEW PATH":
                movement.update()
            elif message == "NEW MOVEMENT":
                movement = action.pop(0)

        data = [item for item in robot_info]

        pose = np.array(data[:6])
        gen_v = np.array(data[6:12])
        f_ext = np.array(data[12:])
        time_of_computation = time.time()
        expected_runtime = movement.estimate_runtime(pose, gen_v, np.linalg.norm(f_ext)) + movement.remaining_runtime()
        path_runtime[:] = [time_of_computation, expected_runtime]


if __name__ == "__main__":
    path_runtime = mp.Array("d", 2)
    update_path_runtime(path_runtime)

