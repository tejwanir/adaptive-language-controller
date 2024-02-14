import numpy as np

from paths import ArkPath, LinePath

# Path Constants
start_pose = np.array(
    [-0.350, -0.6206, 0.04, 0.1671, -1.971, 0.2250]
)  # The starting position for joints
center = np.array(
    [-0.400, -0.256, 0]
)  # The center of rotation used for path planning computations
target_angle = np.deg2rad(-40)  # The target angle at which path is considered complete
rhr = False

# Translational Constants

trans_inertia = 40
trans_stiffness = 3 * trans_inertia
trans_dampness = 2 * trans_inertia

# Rotational Constants
inertia_rot = 3  #
stiffness_rot = 4 * inertia_rot
damping_rot = 3 * inertia_rot

# The force which drives the TCP tangent to the path
driving_force = 1 / 2 * trans_inertia

rot_params = (inertia_rot, stiffness_rot, damping_rot)
trans_params = (trans_inertia, trans_stiffness, trans_dampness)


# Defining different movements using the path class
normal_turn = ArkPath(
    trans_params=trans_params,
    rot_params=rot_params,
    driving_force=driving_force,
    start_pose=start_pose,
    center=center,
    target_angle=target_angle,
    rhr=False,
)

end_pose = normal_turn.get_end_pose()

normal_walk = LinePath(
    trans_params=trans_params,
    rot_params=rot_params,
    driving_force=driving_force,
    start_pose=end_pose,
    end_pose=start_pose,
)

slow_turn = ArkPath(
    trans_params=trans_params,
    rot_params=rot_params,
    driving_force=driving_force / 2,
    start_pose=start_pose,
    center=center,
    target_angle=target_angle,
    rhr=False,
)

slow_walk = LinePath(
    trans_params=trans_params,
    rot_params=rot_params,
    driving_force=driving_force / 2,
    start_pose=end_pose,
    end_pose=start_pose,
)
