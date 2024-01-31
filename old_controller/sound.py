import multiprocessing as mp

from threading import Event

import numpy as np
import time
from rubberband_rt import RubberBandStretcher
from typing import List
from phrasing_digraph import WordGraph

def clear_queue(q: "mp.Queue") -> None:
    while q.qsize() != 0:
        q.get()

# Paco: Look up https://docs.python.org/3/library/multiprocessing.html

def audio_processor(audio_speed: "mp.Value[float]", audio_runtime: "mp.Value[float]",
                    speach: List[WordGraph], admit_to_audio: "mp.Queue[str]",
                    audio_to_admit: "mp.Queue[str]", path_runtime: "mp.Value[float]", resistance: "mp.Value[float]"):
    import sounddevice as sd
    import soundfile as sf

    first_phrasing_digraph = speach.pop(0)
    noise_audio, noise_fs = sf.read("/home/ravi/projects/language-control-robotics/audio/noise-1s.wav", dtype='float32')
    phrasing_digraph_vertex = WordGraph("audio/therapy_short_1.wav", False, [first_phrasing_digraph])

    print("first sentence: ", time.time())
    while True:

        phrasing_digraph_vertex = phrasing_digraph_vertex.next_word(path_runtime[1], resistance.value, use_filler = True)

        if phrasing_digraph_vertex is None and not speach:
            print("last sentence finished: ", time.time())
            break

        if phrasing_digraph_vertex is None:
            print("sentence finished: ", time.time())
            audio_to_admit.put("READY")

            while True:
                if admit_to_audio.qsize() != 0:
                    message = admit_to_audio.get()
                    if message == "GO":
                        break
                time.sleep(0.1)

            phrasing_digraph_vertex = speach.pop(0)

        audio = phrasing_digraph_vertex.current_audio()

        frame_rate = audio.frame_rate

        stretcher = RubberBandStretcher(frame_rate, audio.channels)

        buffer = np.empty([audio.channels, 2 ** 16], dtype=np.single)
        buf_idx = 0
        left_over = 0

        src = np.r_[
            np.zeros(
                [stretcher.getPreferredStartPad(), audio.channels], dtype=np.single
            ),
            (
                    np.asarray(audio.get_array_of_samples()).reshape((-1, audio.channels))
                    / 2 ** 15
            ).astype(np.single),
        ].T

        src = np.ascontiguousarray(src)
        src_idx = 0
        src_len = src.shape[1]

        delay = stretcher.getStartDelay()

        def output_callback(output: np.ndarray, frames: int, time, status):

            nonlocal src_idx, buf_idx, left_over, delay
            nonlocal audio_runtime, frame_rate, phrasing_digraph_vertex

            if delay <= 0:
                rem_word_time = (src_len - src_idx) / frame_rate

                audio_runtime.value = (rem_word_time + phrasing_digraph_vertex.exp_rem_len(max(0, path_runtime[1] - rem_word_time), 1))+1

            out_idx = 0
            if left_over < buf_idx:
                out_idx = min(frames, buf_idx - left_over)
                output[:out_idx, :] = buffer[:, left_over: left_over + out_idx].T
                left_over += out_idx
            while out_idx < frames:
                stretcher.setTimeRatio(1 / audio_speed.value)
                samples = stretcher.getSamplesRequired()
                if samples > 0:
                    next_idx = min(src_idx + samples, src_len)
                    stretcher.process(src[:, src_idx:next_idx], next_idx == src_len)
                    src_idx = next_idx
                buf_idx = stretcher.available()
                if buf_idx == -1:
                    raise sd.CallbackStop()
                stretcher.retrieve(buffer[:, :buf_idx])
                if delay > 0:
                    if buf_idx <= delay:
                        delay -= buf_idx
                        continue
                    buffer[:, : buf_idx - delay] = buffer[:, delay:buf_idx]
                    buf_idx -= delay
                    delay = 0
                left_over = min(buf_idx, frames - out_idx)
                output[out_idx: out_idx + left_over, :] = buffer[:, :left_over].T
                out_idx += left_over

        end_event = Event()

        with sd.OutputStream(
                samplerate=audio.frame_rate,
                channels=audio.channels,
                callback=output_callback,
                finished_callback=end_event.set,
        ):
            end_event.wait()
        sd.play(noise_audio, noise_fs)
        sd.wait()


if __name__ == "__main__":
    print("sound worked")