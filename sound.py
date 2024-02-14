import io
import multiprocessing as mp
import signal
import time
from threading import Event

import numpy as np


def clear_queue(q: "mp.Queue") -> None:
    while q.qsize() != 0:
        q.get()


class Audio:
    def __init__(self, data: np.ndarray, frame_rate: int):
        """Initializes an audio object.

        :param data: The audio data, as a 1D or 2D numpy array (if 2D, the
            second dimension is the number of channels). Values should be in the
            range [-1, 1].
        :param frame_rate: The sample/frame rate of the audio data, in Hz.
        """
        self.data = data
        self.frame_rate = frame_rate
        self.channels = 1 if len(data.shape) == 1 else data.shape[1]


def _play_audio(audio: Audio, audio_speed: "mp.Value[float]"):
    """Plays the given audio at the given speed."""
    import sounddevice as sd
    from rubberband_rt import RubberBandStretcher

    frame_rate = audio.frame_rate
    stretcher = RubberBandStretcher(frame_rate, audio.channels)

    buffer = np.empty([audio.channels, 2**16], dtype=np.single)
    buf_idx = 0
    left_over = 0

    preferred_start_pad = stretcher.getPreferredStartPad()
    src = np.r_[
        np.zeros([preferred_start_pad, audio.channels], dtype=np.single),
        audio.data.reshape((-1, audio.channels)).astype(np.single),
    ].T

    src = np.ascontiguousarray(src)
    src_idx = 0
    src_len = src.shape[1]

    delay = stretcher.getStartDelay()

    def output_callback(output: np.ndarray, frames: int, time, status):
        nonlocal src_idx, buf_idx, left_over, delay, frame_rate

        out_idx = 0
        if left_over < buf_idx:
            out_idx = min(frames, buf_idx - left_over)
            output[:out_idx, :] = buffer[:, left_over : left_over + out_idx].T
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
            output[out_idx : out_idx + left_over, :] = buffer[:, :left_over].T
            out_idx += left_over

    end_event = Event()

    with sd.OutputStream(
        samplerate=audio.frame_rate,
        channels=audio.channels,
        callback=output_callback,
        finished_callback=end_event.set,
    ):
        end_event.wait()
    sd.wait()


def _join_interruptible(process: "mp.Process", timeout: float = 0.1) -> None:
    prev_handler = signal.signal(signal.SIGINT, lambda *_: process.terminate())
    while process.is_alive():
        process.join(timeout)
    signal.signal(signal.SIGINT, prev_handler)


def _audio_loop(
    audio_speed: "mp.Value[float]",
    audio_segments: "mp.Queue[Audio]",
):
    while True:
        audio = audio_segments.get()
        _play_audio(audio, audio_speed)


def tts_loop(
    audio_speed: "mp.Value[float]",
    ready: "mp.Event",
    texts: "mp.Queue[str]",
):
    import boto3
    import soundfile as sf
    from botocore.config import Config

    config = Config(region_name="us-east-1")
    client = boto3.client("polly", config=config)
    ready.set()
    while True:
        text = texts.get()
        start = time.time()
        response = client.synthesize_speech(
            Engine="neural",
            LanguageCode="en-US",
            OutputFormat="mp3",
            Text=text,
            TextType="text",
            VoiceId="Joanna",
        )
        output = io.BytesIO(response["AudioStream"].read())
        print(f"Synthesized {text!r} in {time.time() - start:.2f} seconds")
        data, sample_rate = sf.read(output)
        _play_audio(Audio(data, sample_rate), audio_speed)
        ready.set()


class AsyncTTSPlayer:
    def __init__(self):
        self.audio_speed = mp.Value("d", 1.0)
        self.texts: "mp.Queue[str]" = mp.Queue()
        self.ready: "mp.Event" = mp.Event()
        self.process = mp.Process(
            target=tts_loop, args=(self.audio_speed, self.ready, self.texts)
        )
        self.process.start()

    def set_speed(self, speed: float):
        self.audio_speed.value = speed

    def put_text(self, text: str):
        self.texts.put(text)

    def put_text_if_ready(self, text: str):
        if self.ready.is_set():
            self.texts.put(text)
            self.ready.clear()

    def join(self):
        _join_interruptible(self.process)

    def stop(self):
        self.process.terminate()
