import io
import json
import subprocess
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import pyaudio
from PyNuitrack import py_nuitrack

# fmt: off
JOINT_NAMES = [
    "head", "neck", "torso", "waist",
    "left_collar",  "left_shoulder",  "left_elbow",  "left_wrist",  "left_hand",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip",  "left_knee",  "left_ankle",
    "right_hip", "right_knee", "right_ankle",
]
# fmt: on


def draw_skeleton(image, data) -> str:
    point_color = (59, 164, 0)
    frame_data = {}
    frame_data["timestamp"] = time.time()
    frame_data["skeletons"] = []
    for skel in data.skeletons:
        for el in skel[1:]:
            x = (round(el.projection[0]), round(el.projection[1]))
            cv2.circle(image, x, 8, point_color, -1)
        skeleton_data = {"user_id": skel.user_id}
        for joint_name in JOINT_NAMES:
            if not hasattr(skel, joint_name):
                continue
            joint = getattr(skel, joint_name)
            skeleton_data[joint_name] = {
                "confidence": joint.confidence,
                "projection": joint.projection.tolist(),
                "real": joint.real.tolist(),
                "orientation": joint.orientation.tolist(),
            }
        frame_data["skeletons"].append(skeleton_data)
    return json.dumps(frame_data)


def start_mp4_recording(output_path):
    video_device = "Intel(R) RealSense(TM) Depth Camera 435i RGB"
    audio_device = "麦克风 (Realtek(R) Audio)"
    resolution = "1280x720"
    proc = subprocess.Popen(
        [
            # fmt: off
            "ffmpeg",
            "-f", "dshow",
            "-video_size", resolution,
            "-i", f"video={video_device}",
            "-framerate", "30",
            "-c:v", "h264_nvenc",
            "-y", output_path,
            # fmt: on
        ],
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    wrapper = io.TextIOWrapper(proc.stderr, encoding="utf-8")
    for line in wrapper:
        line = line.rstrip()
        if line.startswith("frame="):
            break
        print(line)

    def reader_thread():
        while True:
            line = wrapper.readline()
            if not line:
                break

    threading.Thread(target=reader_thread).start()
    return proc


def main():
    nuitrack = py_nuitrack.Nuitrack()
    nuitrack.init()

    # ---enable if you want to use face tracking---
    nuitrack.set_config_value("CnnDetectionModule.ToUse", "true")

    devices = nuitrack.get_device_list()
    for i, dev in enumerate(devices):
        if i == 0:
            nuitrack.set_device(dev)
            print("set device: ", dev.get_name(), dev.get_serial_number())
            break

    print(nuitrack.get_license())

    nuitrack.create_modules()
    nuitrack.run()

    session_name = "test_session"
    root_path = Path() / "sessions" / session_name
    if not root_path.exists():
        root_path.mkdir(parents=True)
    json_filename = root_path / "poses.jsonl"
    video_filename = root_path / "video.avi"
    audio_filename = root_path / "audio.wav"

    # Setup video recording
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(video_filename), fourcc, 30.0, (640, 480))

    # Setup audio recording
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    stream = pyaudio.PyAudio().open(
        format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = []  # Initialize array to store audio frames

    def record_audio():
        try:
            while True:
                data = stream.read(chunk)
                frames.append(data)
        except OSError:
            pass

    # Start recording audio
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    # Clear data.json
    with open(json_filename, "w") as file:
        file.write("")

    # Start recording stuffs
    with open(json_filename, "a") as file:
        # proc = start_mp4_recording(video_filename)
        while True:
            key = cv2.waitKey(1)
            nuitrack.update()
            data = nuitrack.get_skeleton()
            img_depth = nuitrack.get_depth_data()
            img_color = nuitrack.get_color_data()
            out.write(img_color)
            if img_depth.size:
                cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)
                img_depth = np.array(
                    cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB), dtype=np.uint8
                )
                json = draw_skeleton(img_depth, data)
                cv2.imshow("Image", img_depth)
                file.write(json + "\n")
            if key == 27:
                break
        # proc.send_signal(signal.CTRL_BREAK_EVENT)
        # Stop recording audio
    stream.stop_stream()
    stream.close()

    # Save audio to a WAV file
    with wave.open(str(audio_filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))

    out.release()
    cv2.destroyAllWindows()
    nuitrack.release()


if __name__ == "__main__":
    main()
