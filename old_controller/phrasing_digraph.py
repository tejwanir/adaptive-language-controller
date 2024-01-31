import pydub
import random
import math


class WordGraph:

    def __init__(self, file_name: str, can_end: bool, children=None, is_filler=False):
        """
        @param file_name: The name of the  audio file ( as a string)
        @param can_end: True if the sentence could be ended there, False if not
        @param children: The possible words that could be said after
        """

        self.children = children
        self.audio = pydub.AudioSegment.from_wav(file_name)
        self.audio_len = (self.audio.frame_count() / self.audio.frame_rate) + 1
        self.can_end = can_end

        self.min_len = self.audio_len
        self.max_len = self.audio_len
        self.is_filler = is_filler

        if children:
            self.min_len += min(child.min_len for child in children)
            self.max_len += max(child.max_len for child in children)

    def add_children(self, new_children):
        if not new_children:
            return

        if self.children:
            self.children.extend(new_children)
        else:
            self.children = new_children[:]

    def get_center(self):
        return (self.max_len + self.min_len) / 2

    def get_min(self):
        return self.min_len

    def get_max(self):
        return self.max_len

    def set_range(self, range_tuple):
        self.min_len = range_tuple[0]
        self.max_len = range_tuple[1]

    def range_dist(self, path_len: float, resistance: float):
        """
        @param path_len: The length of the expected path
        @param resistance: a float used to represent the users resistance as a scalar ranging from 0 to 1
        @return: A float representing the minimum distance to be outside fo the range of possible values (negative if already outside)
        """
        midpoint = (self.min_len + self.max_len) / 2
        ref_point = resistance * self.max_len + (1 - resistance) * midpoint
        return abs(ref_point - path_len)

    def next_word(self, trajectory_remaining_time: float, resistance: float, use_filler=False):
        """
        @param resistance: a float in the range 0 to 1 describing user resistance
        @param trajectory_remaining_time: the expected path_length at the time of the decision
        @return: The next word ( or sequence of words) that will be spoken by the robot
        """
        if self.children is None:
            return None

        new_path_len = trajectory_remaining_time

        best_child = None
        best_val = float('inf')

        for child in self.children:
            if child.range_dist(new_path_len, resistance) < best_val:
                best_val = child.range_dist(new_path_len, resistance)
                best_child = child

        if self.can_end and trajectory_remaining_time < best_child.min_len:
            return None

        return best_child

    def current_audio(self):
        return self.audio

    def exp_rem_len(self, path_len, resistance: float):
        """
        @param resistance: a float representing user cooperation ( 1 = highest, 0 = lowest0
        @param path_len: The expected path_time at the time of calculation
        @return: the expected time until the sentence is finished
        """

        total_len = 0
        word_graph = self.next_word(path_len, 1, use_filler=False)

        while word_graph is not None:
            total_len += word_graph.audio_len

            word_graph = word_graph.next_word(path_len - total_len, resistance, use_filler=False)

        return total_len


if __name__ == "__main__":
    test_1 = WordGraph("audio/therapy_short_1.wav", True, None)
    # test_2 = WordGraph("audio/therapy_short_2.wav", True, [test_1])
    # test_3 = WordGraph("audio/therapy_short_3.wav", False, [test_1, test_2])
    #
    # test_dict = { "audio/therapy_short_3.wav": ["audio/therapy_short_2.wav","audio/therapy_short_1.wav"],
    #               "audio/therapy_short_2.wav": ["audio/therapy_short_1.wav"],
    #               "audio/therapy_short_1.wav": []}
    #
    # x = build_word_tree(test_dict,"audio/therapy_short_3.wav")
    # print("checking the max")
    #
    # print(x.get_min())
    # print(test_3.get_min())
