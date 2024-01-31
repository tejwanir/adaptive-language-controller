from typing import Tuple, List
from scipy.signal import butter, filtfilt
import numpy as np


# Use filter to avoid noise due to faulty or noisy robot sensor data
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


b, a = butter_lowpass(cutoff=10, fs=500, order=2)

# Define the cutoff frequency and sampling rate
cutoff_freq = 100  # Cutoff frequency in Hz
sampling_rate = 500  # Sampling rate in Hz


class LanguageController:

    def __init__(self, audio_pace_range: Tuple[float, float], physical_pace_range: Tuple[float, float],
                 spring_constants: Tuple,
                 window_len: int) -> None:
        """
       @param audio_pace_range: A tuple containing the min/max audio speeds permitted
       @param physical_pace_range: A tuple containing the min/max path speeds permitted
       @param spring_constants: The constants used in calculating the harmonic control of audio/path_speed
       @param window_len: The number of samples used in the context window for the audio modulation process
       """

        self.audio_min = audio_pace_range[0]
        self.audio_pace_max = audio_pace_range[1]

        self.physical_pace_min = physical_pace_range[0]
        self.physical_pace_max = physical_pace_range[1]

        self.window_len = window_len
        self.context_window = [1] * window_len

        self.physical_pace_constant = spring_constants[0]
        self.audio_pace_constant = spring_constants[1]
        self.dt = 1 / 500

        buffer_size = 500
        self.audio_pace_buffer = np.ones(buffer_size)
        self.physical_pace_buffer = np.ones(buffer_size)

    def harmonic_control(self, audio_len: float, audio_pace: float, path_len: float,
                         physical_pace: float, resistance: float) -> Tuple[float, float, float, float]:
        """
       @param audio_len: the length of audio (in seconds) which still needs to be played
       @param audio_pace: the current audio speed ( as a percentage of the default speed)
       @param path_len: The total time length of the paths that has yet to be completed
       @param physical_pace: the current speed at which the path is being completed  (as a percentage of the normal speed)
       @return: A tuple containing the new audio/path speeds for the next time step
       """

        k_1 = self.physical_pace_constant
        k_2 = self.audio_pace_constant
        dt = self.dt

        resistance_constant = 10

        # Finds ideal values for audio/physical pace. If statement avoids division by 0
        if audio_len != 0:
            k = path_len / audio_len
            ideal_audio_pace = (k + 1) / (k ** 2 + 1)
            ideal_physical_pace = k * ideal_audio_pace
        else:
            ideal_physical_pace = (audio_len / path_len + 1) / ((audio_len / path_len) ** 2 + 1)
            ideal_audio_pace = ((audio_len / path_len) ** 2 + audio_len / path_len) / ((audio_len / path_len) ** 2 + 1)

        physical_pace_derivative = k_1 * (ideal_physical_pace - physical_pace)
        audio_pace_derivative = k_2 * (ideal_audio_pace - audio_pace) - resistance_constant * (resistance)

        audio_pace = audio_pace + audio_pace_derivative * dt
        physical_pace = physical_pace + physical_pace_derivative * dt

        audio_pace = min(self.audio_pace_max, max(audio_pace, self.audio_min))
        physical_pace = min(self.physical_pace_max, max(physical_pace, self.physical_pace_min))

        self.audio_pace_buffer[:-1] = self.audio_pace_buffer[1:]
        self.physical_pace_buffer[:-1] = self.physical_pace_buffer[1:]

        # Add the new sample to the end of the buffer
        self.audio_pace_buffer[-1] = audio_pace
        self.physical_pace_buffer[-1] = physical_pace

        # Applies filter to reduce noise
        filtered_audio_pace = filtfilt(b, a, self.audio_pace_buffer)[-1]
        filtered_physical_pace = filtfilt(b, a, self.physical_pace_buffer)[-1]

        return audio_pace, physical_pace, filtered_audio_pace, filtered_physical_pace
