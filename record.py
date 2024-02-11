import io
import json
import subprocess
import threading
import time
import wave
from pathlib import Path
import serial
import csv
from datetime import datetime

import cv2
import numpy as np
import pyaudio
#from PyNuitrack import py_nuitrack

JOINT_NAMES = [
    "Nose", "Neck", "ShoulderRight", "ElbowRight", "WristRight", "ShoulderLeft", "ElbowLeft", "WristLeft",
    "HipRight", "KneeRight", "AnkleRight", "HipLeft", "KneeLeft", "AnkleLeft", "EyeRight", "EyeLeft",
    "EarRight", "EarLeft", "FootLeft", "FootRight", "Pelvis", "Waist", "Chest", "BigToeLeft", "LittleToeLeft",
    "HeelLeft", "BigToeRight", "LittleToeRight", "HeelRight", "TopSkull", "BackSkull", "Xiphoid",
    "ClavicleLeft", "ClavicleRight", "Head"
]

JOINT_CONNECTIONS = [
    ("Nose", "Neck"),
    ("Neck", "ShoulderRight"),
    ("ShoulderRight", "ElbowRight"),
    ("ElbowRight", "WristRight"),
    ("ShoulderRight", "HipRight"),
    ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"),
    ("Neck", "ShoulderLeft"),
    ("ShoulderLeft", "ElbowLeft"),
    ("ElbowLeft", "WristLeft"),
    ("ShoulderLeft", "HipLeft"),
    ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"),
    ("Nose", "EyeRight"),
    ("Nose", "EyeLeft"),
    ("EyeRight", "EarRight"),
    ("EyeLeft", "EarLeft"),
    ("AnkleLeft", "FootLeft"),
    ("AnkleRight", "FootRight"),
    ("Waist", "Pelvis"),
    ("Chest", "Pelvis"),
    ("TopSkull", "BackSkull"),
    ("Chest", "Xiphoid"),
    ("Xiphoid", "Pelvis"),
    ("ShoulderLeft", "ClavicleLeft"),
    ("ShoulderRight", "ClavicleRight"),
    ("Neck", "Head")
]

'''
# fmt: off
JOINT_NAMES = [
    "head", "neck", "torso", "waist",
    "left_collar",  "left_shoulder",  "left_elbow", "left_hand",
    #"left_wrist",  
    "right_collar", "right_shoulder", "right_elbow", "right_hand",
    #"right_wrist", 
    "left_hip",  "left_knee",  "left_ankle",
    "right_hip", "right_knee", "right_ankle",
]

JOINT_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "torso"),
    ("torso", "waist"),
    ("waist", "left_hip"),
    ("waist", "right_hip"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    #("left_elbow", "left_wrist"),
    #("left_wrist", "left_hand"),
    ("left_elbow", "left_hand"), #change due to wrist not being accurate
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    #("right_elbow", "right_wrist"),
    #("right_wrist", "right_hand"),
    ("right_elbow", "right_hand"), #change due to wrist not being accurate
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
# fmt: on
'''



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

def read_data(ser, writer):
    writer.writerow(['timestamp', 'reading'])

    while True:
        # Read data from Serial until \n (new line) received
        ser_bytes = ser.readline()

        print(ser_bytes)

        # Convert received bytes to text format
        decoded_bytes = ser_bytes[0:len(ser_bytes)-2].decode("utf-8")
        print(decoded_bytes)

        # Retrieve current time
        c = datetime.now()
        current_time = c.strftime('%H:%M:%S')
        print(current_time)

        # If Arduino has sent a string "stop", exit loop
        if decoded_bytes == "stop":
            break

        # Write received data to CSV file
        writer.writerow([time.time(), decoded_bytes])

def main():
    session_name = "lab_session_8" #CHANGE SESSION NAME


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
    channels = 1
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

    
    # Open the serial port and CSV file
    ser = serial.Serial("COM5")
    ser.flushInput()

    csv_file = open('./sessions/'+session_name+'/data.csv',mode='a')
    #csv_file = open('rich1.csv',mode='a')
    csv_writer = csv.writer(csv_file, delimiter=",", escapechar=' ', quoting=csv.QUOTE_NONE)

    # Write out a single character encoded in utf-8; this is default encoding for Arduino serial comms
    ser.write(bytes('x', 'utf-8'))

    # Create a thread for reading data
    read_data_thread = threading.Thread(target=read_data, args=(ser, csv_writer))
    # Start the thread
    read_data_thread.start()
    
    # Clear data.json
    with open(json_filename, "w") as file:
        file.write("")

    # Start recording stuffs
    with open(json_filename, "a") as file:
        # proc = start_mp4_recording(video_filename)
        while True:
            key = cv2.waitKey(1)
            if img_depth.size:
                cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)
                img_depth = np.array(
                    cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB), dtype=np.uint8
                )
                cv2.imshow("Image", img_depth)
                file.write(json + "\n")
            if key == 27:
                break
        # proc.send_signal(signal.CTRL_BREAK_EVENT)
        # Stop recording audio
    stream.stop_stream()
    stream.close()

    out.release()
    cv2.destroyAllWindows()
    print("Video saved")
    print("Nuitrack tracking module released")
    ser.close()
    csv_file.close()
    print("Logging finished")

    # Save audio to a WAV file
    with wave.open(str(audio_filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
    print("Audio saved")

if __name__ == "__main__":
    main()
