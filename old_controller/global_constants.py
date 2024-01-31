from ur5 import dt
import numpy as np
from paths import Path, LinePath, ArkPath
from language_control import LanguageController
from resistance import ResistanceTracker
from phrasing_digraph import WordGraph


# Path Constants
start_pose = np.array([-0.350, -0.6206, 0.04, 0.1671, -1.971, 0.2250])  # The starting position for joints
center = np.array([-0.400, -0.256, 0])  # The center of rotation used for path planning computations
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
driving_force = 1/2*trans_inertia

rot_params = (inertia_rot, stiffness_rot, damping_rot)
trans_params = (trans_inertia, trans_stiffness, trans_dampness)


#Defining different movements using the path class
normal_turn = ArkPath(trans_params=trans_params, rot_params=rot_params, driving_force=driving_force,
                 start_pose=start_pose, center=center, target_angle=target_angle, rhr=False)

end_pose = normal_turn.get_end_pose()

normal_walk = LinePath(trans_params=trans_params, rot_params=rot_params, driving_force=driving_force,
                  start_pose=end_pose, end_pose=start_pose)

slow_turn = ArkPath(trans_params=trans_params, rot_params=rot_params, driving_force=driving_force/2,
                 start_pose=start_pose, center=center, target_angle=target_angle, rhr=False)

slow_walk = LinePath(trans_params=trans_params, rot_params=rot_params, driving_force=driving_force/2,
                  start_pose=end_pose, end_pose=start_pose)


# Audio pace and physical pace modulation constants
ideal_resistance = 0  # The force which an ideally compliant user would exert
window_len = int(0.25 * 1 / dt)
audio_range = (0.6, 1.4)
physical_pace_range = (0.7, 1.4)
modulating_spring_const = (6, 7)


test_lang_controller = LanguageController(audio_pace_range=audio_range, physical_pace_range=physical_pace_range,
                                          spring_constants=modulating_spring_const, window_len=window_len)

test_force_controller = ResistanceTracker(window_len=500, sensitivity=0.5, break_point=20)


#   WORD TREE CONSTRUCTION

your_done = WordGraph("audio/your_done.wav", True, None)
move_back_2_long = WordGraph("audio/move_back_2_long.wav", True, [your_done])
move_back_2_short = WordGraph("audio/move_back_2_short.wav", True, [your_done])
move_out_2 = WordGraph("audio/move_out_2.wav", False, [move_back_2_long, move_back_2_short])
desc_1 = WordGraph("audio/like_washing_table.wav", False, [move_out_2])
move_back_1 = WordGraph("audio/move_back_1.wav", False, [move_out_2, desc_1])
move_out_1 = WordGraph("audio/move_out_1.wav", False, [move_back_1])


#   TREE 1 CONSTRUCTION
tree_1_taking_as_long = WordGraph("audio/tree_1_taking_as_long.wav", True, None)
tree_1_slowly_extend = WordGraph("audio/tree_1_slowly_extend.wav", True, [tree_1_taking_as_long])
tree_1_move_hand_out = WordGraph("audio/tree_1_move_hand_out.wav", True, [tree_1_taking_as_long])
tree_1_root = WordGraph("audio/tree_1_I_want_you_to.wav", False, [tree_1_move_hand_out, tree_1_slowly_extend])

# TREE 2 CONSTRUCTION
tree_2_washing_table = WordGraph("audio/tree_2_kind_of_like_washing_table.wav", True, None)
tree_2_move_backwards = WordGraph("audio/tree_2_move_hand_backwards.wav", True, [tree_2_washing_table])
tree_2_bring_back = WordGraph("audio/tree_2_bring_it_back.wav", True, [tree_2_washing_table])
tree_2_I_want = WordGraph("audio/tree_1_I_want_you_to.wav", False, [tree_2_bring_back, tree_2_move_backwards])
tree_2_root = WordGraph("audio/tree_2_now.wav", False, [tree_2_bring_back, tree_2_move_backwards, tree_2_I_want])

# TREE 3 CONSTRUCTION
tree_3_once_more_time = WordGraph("audio/tree_3_one_more_time.wav", True, None)
tree_3_extend_arm = WordGraph("audio/tree_3_extend_arm.wav", True, [tree_3_once_more_time])
tree_3_move_forwards = WordGraph("audio/tree_3_move_forwards.wav", True, [tree_3_once_more_time])
tree_3_I_want = WordGraph("audio/tree_3_move_forwards.wav", False, [tree_3_extend_arm, tree_3_move_forwards])
tree_3_root = WordGraph("audio/tree_3_now.wav", False, [tree_3_extend_arm, tree_3_move_forwards, tree_3_I_want])

#TREE 4 CONSTRUCTION

tree_4_your_done = WordGraph("audio/tree_4_and_your_done.wav", True, None)
tree_4_move_backwards = WordGraph("audio/tree_4_move_hand_backwards.wav", True, [tree_4_your_done])
tree_4_move_hand_back = WordGraph("audio/tree_4_move_your_hand_back.wav", True, [tree_4_your_done])
tree_4_root = WordGraph("audio/tree_4_I_want_you.wav", False, [tree_4_move_hand_back, tree_4_move_backwards])

#DEMO TREE CONSTRUCTION

demo_tree_short_end = WordGraph("audio/demo_tree_short_ending.wav", True, None)
demo_tree_long_end = WordGraph("audio/demo_tree_long_end.wav", True, None)
demo_tree_conversion_point = WordGraph("audio/demo_tree_branch_conversion.wav", False, [demo_tree_long_end,
                                                                                        demo_tree_short_end])
demo_tree_short_continuation = WordGraph("audio/demo_tree_short_continuation.wav", False, [demo_tree_conversion_point])
demo_tree_long_continuation = WordGraph("audio/demo_tree_longer_continuation.wav", False, [demo_tree_conversion_point])

demo_tree_short_branch = WordGraph("audio/demo_tree_short_branch.wav", False, [demo_tree_short_continuation,
                                                                               demo_tree_long_continuation])
demo_tree_medium_branch = WordGraph("audio/demo_tree_medium_branch.wav", False, [demo_tree_short_continuation,
                                                                                 demo_tree_long_continuation])
demo_tree_long_branch = WordGraph("audio/demo_tree_longest_branch.wav", False, [demo_tree_short_continuation,
                                                                                demo_tree_long_continuation])
demo_tree_detour = WordGraph("audio/demo_tree_detour.wav", False, [demo_tree_short_branch, demo_tree_long_branch,
                                                                   demo_tree_medium_branch])
demo_tree_root = WordGraph("audio/demo_tree_start.wav", False, [demo_tree_short_branch, demo_tree_long_branch,
                                                                demo_tree_medium_branch, demo_tree_detour])


m1_one_more_good_one = WordGraph("audio/Medium Resistance/One more good one-m1.wav", True, None)
m1_try_to_relax_those_fingers = WordGraph("audio/Medium Resistance/Try to relax those fingers alright-m1.wav", False, [m1_one_more_good_one])
m1_now_glide_over = WordGraph("audio/Medium Resistance/Glide over-m1.wav", False, [m1_one_more_good_one, m1_try_to_relax_those_fingers ])
m1_glide_real_smoothly = WordGraph("audio/Medium Resistance/Glide over-m1.wav", False, [m1_now_glide_over])
m1_glide_like_you_are_iceskating = WordGraph("audio/Medium Resistance/Glide like you are ice skating on this table-m1.wav", False, [m1_now_glide_over])
m1_head = WordGraph("audio/Medium Resistance/Glide towards the edge of the table ok-m1.wav", False, [m1_now_glide_over, m1_glide_like_you_are_iceskating, m1_glide_real_smoothly ])


#layer 1
test_back_to_middle = WordGraph("audio/Medium Resistance/and I'm gonna bring you back to the middle-m5.wav", True, None)
#test_and_move_back_to_middle = WordGraph("audio/Medium Resistance/and move back to the middle-m5.wav", True, None)
test_and_to_middle = WordGraph("audio/Medium Resistance/and to the middle-m5.wav", True, None)

#layer 2
test_good_way_back = WordGraph("audio/Medium Resistance/good and nice and easy on the way back-m4.wav", False, None)
test_everything_way_back = WordGraph("audio/Medium Resistance/let me do everything on the way back-m3.wav", False, None)

#layer 3
test_glide_real_smooth = WordGraph("audio/Medium Resistance/Glide really smoothly-m1-faster.wav", False, [test_good_way_back, test_everything_way_back, test_and_to_middle])
test_we_are_across_table = WordGraph("audio/High Resistance/we are sliding across the table together-h2.wav", False, [test_good_way_back, test_everything_way_back,  test_back_to_middle ])
test_relax_fingers = WordGraph("audio/Medium Resistance/Try to relax those fingers alright-m1.wav", False, [test_good_way_back, test_everything_way_back ])
test_help_2 = WordGraph("audio/Medium Resistance/let me do everything on the way back-m3.wav", True, None)
test_help_3 = WordGraph("audio/Medium Resistance/Glide really smoothly-m1-faster.wav", True, None)

#layer 4x
test_help_1 = WordGraph("audio/Medium Resistance/Glide really smoothly-m1-faster.wav", True, None)
test_glide_like_ice_skating = WordGraph("audio/Medium Resistance/Glide like you are ice skating on this table-m1.wav", False, [test_help_3, test_help_2, test_glide_real_smooth, test_we_are_across_table, test_relax_fingers, test_and_to_middle])
test_slide_slide_slide = WordGraph("audio/High Resistance/slide slide slide-h2.wav", False, [test_help_3 , test_help_2, test_glide_real_smooth, test_good_way_back, test_we_are_across_table, test_relax_fingers])
test_to_the_side_here = WordGraph("audio/Medium Resistance/and to the side here-m5.wav", False, [test_help_3 ,test_help_2,test_we_are_across_table, test_relax_fingers])
#test_very_long_sentence = WordGraph("audio/big a", False, [test_glide_real_smooth, test_we_are_across_table, test_relax_fingers])

#
test_head = WordGraph("audio/High Resistance/i want you to move your hand away from your body this way-h1.wav", False, [test_slide_slide_slide,  test_help_1, test_glide_like_ice_skating])


baseline_head = WordGraph("audio/shorter_big_audio.wav", True, None)
