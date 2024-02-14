from typing import Tuple, Callable, List, Any

import numpy as np
from utils import angle_between, rotate_pose, sim_speedL
from scipy.spatial.transform import Rotation as R
import time


class Path:
    def __init__(self):
        self.driving_force = None
        self.start_pose = None
        self.trans_inertia = None
        self.rot_inertia = None
        self.trans_damping = None
        self.trans_stiffness = None

    def force_torque(self):
        pass

    def get_end_pose(self):
        pass

    def get_driving_force(self):
        return self.driving_force

    def set_constants(self, new_driving_force, new_trans_damping, new_trans_stiffness):
        self.trans_damping = new_trans_damping
        self.trans_stiffness = new_trans_stiffness
        self.driving_force = new_driving_force

    def find_v_ideal(self, pose: np.array, velocity: np.array, omega: np.array) -> np.array:
        """
       @param pose: The pose of the robot
       param velocity: the linear velocity of the robot
       @param omega: the angular velocity of the robot
       Calculates ideal velocity with equation from paper
       """
        trans_damping = self.trans_damping

        damping_matrix = np.diag([trans_damping] * 3)
        dampening_inv = np.linalg.inv(damping_matrix)

        ideal_resistance = 0.1

        (f_drive, t_drive) = self.force_torque(pose, velocity, omega)
        f_ideal = - ideal_resistance * f_drive  # ideal resistance force
        v_ideal_vec = (f_drive + f_ideal) @ dampening_inv
        return v_ideal_vec

    def estimate_runtime(self, init_pose: np.ndarray, init_v: np.ndarray, f_ext_norm: float) -> float:
        """
       @param init_pose: The initial pose from which the simulation will start
       @param init_v: The initial TCP velocity from which the simulation will start
       @return: The time it took to reach the target position in the simulation
       """

        rot_inertia = self.rot_inertia
        trans_inertia = self.trans_inertia
        end_pose = self.get_end_pose()

        pos_epsilon = 0.06
        sim_dt = 1 / 100

        run_time = 0

        pose = init_pose

        v = init_v[:3]
        omega = init_v[3:]
        gen_v = init_v

        initial_delta = pose - end_pose
        start = time.time()
        while True:
            if time.time() - start > 0.2:
                break
            if np.linalg.norm(pose - end_pose) <= pos_epsilon:
                # np.linalg.norm(gen_v) <= speed_epsilon:
                break

            f_virtual, tau_virtual = self.force_torque(pose, v, omega)
            # rate = max(0.1, 1 - f_ext_norm / np.linalg.norm(f_virtual))
            rate = 1
            f_total = rate * f_virtual
            tau_total = rate * tau_virtual
            next_v = v + sim_dt * f_total / trans_inertia
            next_omega = omega + sim_dt * tau_total / rot_inertia
            next_gen_v = np.concatenate((next_v, next_omega))
            pose, next_v_ = sim_speedL(pose, gen_v, next_gen_v, 2, sim_dt)
            v = next_v_[:3]
            omega = next_v_[3:]
            gen_v = next_gen_v
            run_time += sim_dt

        # print(run_time, initial_delta, init_v)
        return run_time

    def get_start_pose(self):
        """
       @return: The starting pose of the path
       """
        return self.start_pose

    def get_trans_inertia(self) -> float:
        """
       @return: The inertia of the virtual object under translational motion
       """
        return self.trans_inertia

    def get_rot_inertia(self):
        """
       @return: The inertia of the virtual object under rotational motion
       """
        return self.rot_inertia

    def total_runtime(self) -> float:
        """
       @return: The expected runtime of the path given that the robot starts/ends in the
       expected positions and the user exerts the ideal amount of resistance
       """

        expected_init_v = np.zeros(6)
        expected_pose = self.start_pose

        runtime_estimate = self.estimate_runtime(expected_pose, expected_init_v, 0)
        return runtime_estimate


class ArkPath(Path):
    """
   Used to create and deal with admittance controllers for ark like paths
   """

    def __init__(self, trans_params: Tuple[float, float, float], rot_params: Tuple[float, float, float],
                 driving_force: float,
                 start_pose: np.ndarray, center: np.ndarray, target_angle: float, rhr: bool):
        """
       @param trans_params: A tuple containing the stiffness and dampness constants for linear motion
       @param rot_params:  A tuple containing the stiffness and dampness constants for Rotational motion
       @param start_pose: The pose from which the robot will be starting
       @param center: The center around which the rotational motion will take place
       @param target_angle: The angle (in degrees) to which the robot should rotate to
       @param rhr: True if the rotation follows the RHR wrt center and False otherwise
       """
        self.trans_inertia = trans_params[0]
        self.trans_stiffness = trans_params[1]
        self.trans_damping = trans_params[2]

        self.rot_inertia = rot_params[0]
        self.rot_stiffness = rot_params[1]
        self.rot_damping = rot_params[2]

        self.driving_force = driving_force

        self.target = target_angle
        self.rhr = rhr
        self.start_pose = start_pose
        self.center = center

        if self.rhr:
            self.end_pose = rotate_pose(self.start_pose, self.center, [0, 0, -self.target])
        else:
            self.end_pose = rotate_pose(self.start_pose, self.center, [0, 0, self.target])

    def path_fidelity(self, pose: np.ndarray) -> float:

        rhr = self.rhr
        target_angle = self.target
        center = self.center

        initial_pose = self.start_pose
        end_pose = self.end_pose

        if rhr:
            lead = 0.01
        else:
            lead = -0.01

        initial_trans = initial_pose[:3]

        trans = pose[:3]
        rot = pose[3:]

        angle_now = angle_between(trans, initial_trans, center)

        ideal_current_pose = rotate_pose(initial_pose, center, [0, 0, angle_now])

        if (not rhr and angle_now < target_angle) or (rhr and angle_now > target_angle):
            ideal_current_pose = end_pose

        return np.linalg.norm(pose[:3] - ideal_current_pose[:3])

    def force_torque(self, pose: np.ndarray, v: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
       @param pose: the current pose of the robot
       @param v: the current velocity of the TCP in linear space
       @param omega: the current angular velocity of the TCP
       @return: A Tuple containing the virtual force and virtual torque to be used in the admittance control loop
       """

        k_1 = self.trans_stiffness
        d_1 = self.trans_damping

        k_2 = self.rot_stiffness
        d_2 = self.rot_damping

        driving_force = self.driving_force

        rhr = self.rhr
        target_angle = self.target
        center = self.center

        initial_pose = self.start_pose
        end_pose = self.end_pose

        if rhr:
            lead = 0.01
        else:
            lead = -0.01

        initial_trans = initial_pose[:3]

        trans = pose[:3]
        rot = pose[3:]

        angle_now = angle_between(trans, initial_trans, center)

        ideal_current_pose = rotate_pose(initial_pose, center, [0, 0, angle_now])
        ideal_next_pose = rotate_pose(initial_pose, center, [0, 0, angle_now + lead])
        f_tangent = ideal_next_pose[:3] - ideal_current_pose[:3]

        f_tangent = driving_force * f_tangent / np.linalg.norm(f_tangent)

        f_radial = k_1 * (ideal_current_pose[:3] - trans)

        f = f_tangent + f_radial

        if (not rhr and angle_now < target_angle) or (rhr and angle_now > target_angle):
            ideal_current_pose = end_pose
            f = k_1 * (ideal_current_pose[:3] - trans)

            # introduce damping factors
        f = f - d_1 * v
        tau = k_2 * (R.from_rotvec(ideal_current_pose[3:]) * R.from_rotvec(rot).inv()).as_rotvec() - d_2 * omega

        return f, tau

    def get_end_pose(self) -> np.ndarray:
        """
       @return: The pose fo the robot after the path has been completed
       """
        return self.end_pose


class LinePath(Path):
    """
   Used to create linear paths
   """

    def __init__(self, trans_params: Tuple[float, float, float], rot_params: Tuple[float, float, float],
                 driving_force: float, start_pose: np.ndarray, end_pose: np.ndarray):
        """
       @param trans_params: A tuple containing the stiffness and dampness constants for linear motion
       @param rot_params:  A tuple containing the stiffness and dampness constants for Rotational motion
       @param start_pose: The pose from which the robot will be starting
       @param end_pose: The pose at which the path will end
       """

        self.trans_inertia = trans_params[0]
        self.trans_stiffness = trans_params[1]
        self.trans_damping = trans_params[2]

        self.rot_inertia = rot_params[0]
        self.rot_stiffness = rot_params[1]
        self.rot_damping = rot_params[2]

        self.driving_force = driving_force

        self.start_pose = start_pose
        self.target = end_pose

    def get_end_pose(self) -> np.ndarray:
        return self.target


    def path_fidelity(self, pose: np.ndarray) -> float:
        end_trans = self.target[:3]

        start_trans = self.start_pose[:3]

        dir_vec = end_trans - start_trans

        length = np.linalg.norm(dir_vec)
        dir_vec = dir_vec / length

        trans = pose[:3]
        trans_diff = trans - start_trans

        t = np.dot(trans_diff, dir_vec) / length
        ideal_trans = start_trans * (1 - t) + end_trans * t

        return np.linalg.norm(pose[:3] - ideal_trans)
    def force_torque(self, pose: np.ndarray, v: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
       @param pose: the current pose of the robot
       @param v: the current velocity of the TCP in linear space
       @param omega: the current angular velocity of the TCP
       @return: A Tuple containing the virtual force and virtual torque to be used in the admittance control loop
       """

        d_1 = self.trans_damping
        k_1 = self.trans_stiffness

        d_2 = self.rot_damping
        k_2 = self.rot_stiffness

        f_drive = self.driving_force

        end_trans = self.target[:3]
        end_rot = self.target[3:]

        start_trans = self.start_pose[:3]
        start_rot = self.start_pose[3:]

        dir_vec = end_trans - start_trans

        length = np.linalg.norm(dir_vec)
        dir_vec = dir_vec / length

        trans = pose[:3]
        trans_diff = trans - start_trans
        rot = pose[3:]

        t = np.dot(trans_diff, dir_vec) / length

        ideal_trans = start_trans * (1 - t) + end_trans * t
        ideal_rot = start_rot * (1 - t) + end_rot * t

        f_tangent = f_drive * dir_vec
        f_radial = -k_1 * (pose[:3] - ideal_trans)

        f = f_tangent + f_radial

        if t > 0.98:
            f = -k_1 * (trans - end_trans)

        f = f - d_1 * v
        tau = k_2 * (R.from_rotvec(ideal_rot) * R.from_rotvec(rot).inv()).as_rotvec() - d_2 * omega

        return f, tau


class Movement:
    """
   Essentially a list of paths plus some functionalities that make code more readable
   """

    def __init__(self, list_of_paths: List[Path]):
        self.paths = list_of_paths[:]
        self.path_times = [path.total_runtime() for path in self.paths]

    def paths_left(self):
        if len(self.paths) <= 1:
            return False
        return True

    def update(self):
        self.paths.pop(0)
        self.path_times.pop(0)

    def remaining_runtime(self):
        return sum(self.path_times[1:])

    def current_path(self):
        return self.paths[0]

    def force_torque(self, pose: np.ndarray, v: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cur_path = self.paths[0]

        return cur_path.force_torque(pose, v, omega, )

    def constants(self):
        cur_path = self.paths[0]
        trans_inertia = cur_path.get_trans_inertia()
        rot_inertia = cur_path.get_rot_inertia()
        driving_force = cur_path.get_driving_force()
        return trans_inertia, rot_inertia, driving_force

    def new_target(self):
        cur_path = self.paths[0]
        end_pose = cur_path.get_end_pose()
        return end_pose

    def start_pose(self):
        cur_path = self.paths[0]
        return cur_path.get_start_pose()

    def end_pose(self):
        cur_path = self.paths[0]
        return cur_path.get_end_pose()

    def estimate_runtime(self, init_pose: np.ndarray, init_v: np.ndarray, f_ext_norm: float):
        cur_path = self.paths[0]
        runtime_estimate = cur_path.estimate_runtime(init_pose, init_v, f_ext_norm)
        return runtime_estimate

    def total_runtime(self):
        return sum(self.path_times)

    def path_fidelity(self, pose: np.ndarray):
        cur_path = self.paths[0]
        #print("path fidelity: ", cur_path.path_fidelity(pose))
        return cur_path.path_fidelity(pose)

