import json
import typing
from pathlib import Path

import numpy as np
from tqdm import tqdm

whisper_model = None
punct_model = None
nlp = None

# Put the Lab Data folder in the same directory as this file


def get_session_base_path_1(session_id: int) -> Path:
    return Path(f"./Lab Data/V1 (post-processed)/Session {session_id + 1}")


def get_session_base_path_2(session_id: int) -> Path:
    return Path(
        f"./Lab Data/V2 (post-processed-and-fixing-errors)/Session {session_id + 1}"
    )


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


def parse_transcription_to_phrases(session_id: int):
    import re

    import spacy
    from deepmultilingualpunctuation import PunctuationModel

    global nlp, punct_model
    if nlp is None:
        nlp = spacy.load("en_core_web_trf")
    if punct_model is None:
        punct_model = PunctuationModel()

    base_path = get_session_base_path_1(session_id)
    with open(base_path / "transcription.json", "r") as f:
        transcription = json.load(f)
    segments = []
    for segment in transcription["segments"]:
        text: str = segment["text"].strip()
        text_fixed = punct_model.restore_punctuation(text)
        phrases = [p for p in re.split(r"[.!?,]", text_fixed) if p]
        next_word_idx = 0
        normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower()).replace("' ", "'")
        for phrase in phrases:
            phrase = normalize(phrase)
            phrase_words = phrase.split()
            phrase_idx = 0
            start_idx = next_word_idx
            while (
                next_word_idx < len(segment["words"])
                and phrase_idx < len(phrase_words)
                and normalize(
                    re.sub(r"[.!?,]", "", segment["words"][next_word_idx]["text"])
                )
                == phrase_words[phrase_idx]
            ):
                next_word_idx += 1
                phrase_idx += 1
            start_time = segment["words"][start_idx]["start"]
            end_time = segment["words"][next_word_idx - 1]["end"]
            segments.append(
                {
                    "phrase": phrase,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
    with open(base_path / "phrases.json", "w") as f:
        json.dump(segments, f, indent=4)


def debug_pose_data(session_id: int):
    """
    This function is just for debugging purposes. DO NOT USE IT.
    """
    import matplotlib.pyplot as plt

    base_path = get_session_base_path_2(session_id)

    with open(base_path / "cut_poses_error_fix.jsonl", "r") as f:
        poses = [json.loads(line) for line in f]

    timestamps = []
    hand_coords = [[], [], [], []]
    last_filtered = None
    last_timestamp = None
    alpha = 0.7
    for pose in poses:
        skeletons = pose["skeletons"]
        if len(skeletons) != 2:
            continue
        hand_coords[0].append(skeletons[0]["left_hand"]["real"])
        right_hand = np.array(skeletons[0]["right_hand"]["real"])
        if last_filtered is not None:
            a = alpha ** (pose["timestamp"] - last_timestamp)
            right_hand = a * last_filtered + (1 - a) * right_hand
        hand_coords[1].append(right_hand)
        hand_coords[2].append(skeletons[1]["left_hand"]["real"])
        hand_coords[3].append(skeletons[1]["right_hand"]["real"])
        timestamps.append(pose["timestamp"])

        last_filtered = right_hand
        last_timestamp = pose["timestamp"]

    timestamps = np.array(timestamps)
    hand_coords = np.array(hand_coords)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    joints = []
    for label in ["0l", "0r", "1l", "1r"]:
        joints.append(ax.plot([], [], [], label=label)[0])
    ax.legend()
    min_x, max_x = hand_coords[:, :, 0].min(), hand_coords[:, :, 0].max()
    ax.set_xlabel("X")
    ax.set_xlim(min_x, max_x)
    min_y, max_y = hand_coords[:, :, 2].min(), hand_coords[:, :, 2].max()
    ax.set_ylim(min_y, max_y)
    ax.set_ylabel("Y")
    min_z, max_z = hand_coords[:, :, 1].min(), hand_coords[:, :, 1].max()
    ax.set_zlim(min_z, max_z)
    ax.set_zlabel("Z")
    fig.show()
    # avg_internal = np.diff(timestamps).mean()

    interval = 50
    n_frames = int((timestamps[-1] - timestamps[0]) * 1000 / interval)
    with tqdm(total=n_frames) as pbar:

        def update(frame):
            timestamp = timestamps[0] + frame * (interval / 1000)
            i1 = np.searchsorted(timestamps, timestamp, side="left")
            i2 = np.searchsorted(timestamps, timestamp - 1, side="right")
            for i in range(4):
                slice = hand_coords[i][i2:i1]
                joints[i].set_data_3d(slice[:, 0], slice[:, 2], slice[:, 1])
            pbar.update(1)
            return tuple(joints)

        for i in range(n_frames):
            update(i)
            plt.pause(interval / 1000)
            plt.draw()


def get_filtered_features(session_id: int, alpha: float = 0.7):
    base_path = get_session_base_path_2(session_id)

    with open(base_path / "cut_poses_error_fix.jsonl", "r") as f:
        poses = [json.loads(line) for line in f]

    timestamps = []
    hand_coords = []
    last_filtered = None
    last_timestamp = None
    for pose in poses:
        skeletons = pose["skeletons"]
        if len(skeletons) != 2:
            continue
        if skeletons[0]["left_hand"]["confidence"] < 0.5:
            continue
        right_hand = np.array(skeletons[0]["right_hand"]["real"])
        if last_filtered is not None:
            a = alpha ** (pose["timestamp"] - last_timestamp)
            right_hand = a * last_filtered + (1 - a) * right_hand
        hand_coords.append(right_hand)
        timestamps.append(pose["timestamp"])
        last_filtered = right_hand
        last_timestamp = pose["timestamp"]
    timestamps = np.array(timestamps)
    hand_coords = np.array(hand_coords)
    return timestamps - timestamps.min(), hand_coords


def adjust_lims(ax, coords):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    fudge = 0.1
    ax.set_xlim(x_min - (x_max - x_min) * fudge, x_max + (x_max - x_min) * fudge)
    ax.set_ylim(y_min - (y_max - y_min) * fudge, y_max + (y_max - y_min) * fudge)
    ax.set_zlim(z_min - (z_max - z_min) * fudge, z_max + (z_max - z_min) * fudge)


def build_knn(session_ids: typing.Iterable[int], plot: bool = False):
    db = []
    for i in session_ids:
        base_path = get_session_base_path_1(i)
        with open(base_path / "phrases.json", "r") as f:
            phrases = json.load(f)
        timestamps, hand_coords = get_filtered_features(i)
        for phrase in phrases:
            start_time = phrase["start_time"]
            t_idx = np.searchsorted(timestamps, start_time, side="left")
            t_idx = min(t_idx, len(hand_coords) - 1)
            s_idx = np.searchsorted(timestamps, start_time - 1, side="left")
            vec = hand_coords[t_idx] - hand_coords[s_idx]
            db.append(
                {
                    "vec": vec.tolist(),
                    "phrase": phrase["phrase"],
                    "duration": phrase["end_time"] - phrase["start_time"],
                }
            )
    with open("knn_db.json", "w") as f:
        json.dump(db, f, indent=4)

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        all_vecs = np.array([item["vec"] for item in db])
        # all_vecs /= np.linalg.norm(all_vecs, axis=1)[:, None] + 1e-6
        adjust_lims(ax, all_vecs)
        ax.scatter(all_vecs[:, 0], all_vecs[:, 1], all_vecs[:, 2])
        for i, item in enumerate(db):
            phrase = item["phrase"]
            ax.text(all_vecs[i, 0], all_vecs[i, 1], all_vecs[i, 2], phrase, fontsize=8)
        plt.show()


def test_knn(session_id: int):
    import time

    import cv2
    from scipy.spatial import KDTree

    timestamps, hand_coords = get_filtered_features(session_id)
    base_path = get_session_base_path_1(session_id)
    cap = cv2.VideoCapture(str(base_path / "cut_video.avi"))
    interval = int(1000 / cap.get(cv2.CAP_PROP_FPS))

    with open("knn_db.json", "r") as f:
        db = json.load(f)
        all_vecs = np.array([item["vec"] for item in db])
        kd_tree = KDTree(all_vecs)

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        t_idx = np.searchsorted(timestamps, t, side="left")
        if t_idx >= len(hand_coords):
            break
        s_idx = np.searchsorted(timestamps, t - 1, side="left")
        vec = hand_coords[t_idx] - hand_coords[s_idx]
        knn_ds, knn_is = kd_tree.query(vec, k=3)
        for i, (knn_d, knn_i) in enumerate(zip(knn_ds, knn_is)):
            item = db[knn_i]
            scale = 0.5 if i == 0 else 0.4
            base_y = [0, 70, 126][i]

            def put_text(content: str, y: float):
                nonlocal frame
                frame = cv2.putText(
                    frame,
                    content,
                    org=(0, int(y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=scale,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            put_text(f"{item['phrase']}", base_y + scale * 40)
            put_text(f"Duration: {item['duration']:.2f}s", base_y + scale * 80)
            put_text(f"Distance: {knn_d:.2f}", base_y + scale * 120)

        cv2.imshow("Video", frame)
        to_wait = max(1, interval - int((time.time() - start) * 1000))
        if cv2.waitKey(to_wait) & 0xFF == 27:
            break


if __name__ == "__main__":
    # Build KNN db from sessions 1-6
    # build_knn(range(1, 7), plot=True)
    # Test KNN on session 0
    test_knn(0)
