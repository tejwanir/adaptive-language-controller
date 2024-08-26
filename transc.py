import json
import multiprocessing as mp
import typing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filter_poses import FrameCorrector, PoseFilter, UserPoses, preprocess_poses
from lightbuzz_poses import collect_poses

whisper_model = None
punct_model = None
nlp = None


def get_session_base_path_1(session_id: int) -> Path:
    return Path(f"./low/lightbuzz_session_low_{session_id + 1}")


def transcribe_session(session_id: int):
    import whisper
    import whisper_timestamped

    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("medium.en")

    base_path = get_session_base_path_1(session_id)
    audio = whisper.load_audio(base_path / "cut_audio.wav")
    output = whisper_timestamped.transcribe(whisper_model, audio, language="en")
    with open(base_path / "transcription.json", "w") as f:
        json.dump(output, f, indent=4)

transcribe_session(0)