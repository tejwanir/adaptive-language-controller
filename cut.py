import json
import pandas as pd
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

# List of sessions with folder names and their corresponding start and end times (s)
sessions = [
    {'folder': 'lightbuzz_table_1', 'start': 33, 'end': 75},
    {'folder': 'lightbuzz_table_2', 'start': 5.5, 'end': 53.5},
    {'folder': 'lightbuzz_table_3', 'start': 4, 'end': 52},
    {'folder': 'lightbuzz_table_4', 'start': 3.5, 'end': 41},
    {'folder': 'lightbuzz_table_5', 'start': 2.5, 'end': 47.5},
    {'folder': 'lightbuzz_table_6', 'start': 5, 'end': 73}
]

for session in sessions:
    folder = session['folder']
    start = session['start']
    end = session['end']

    # Define the paths to your files
    base_path = f'C:/Users/tejwa/adaptive-language-controller/sessions/{folder}'
    poses_path = f'{base_path}/poses.jsonl'
    audio_path = f'{base_path}/audio.wav'
    video_path = f'{base_path}/output.mp4'
    force_sensor_path = f'{base_path}/force_data.csv'

    # Read the poses.jsonl file and extract the first timestamp
    with open(poses_path, 'r') as poses_file:
        poses_data = [json.loads(line) for line in poses_file]
        start_time_unix = poses_data[0]['timestamp']

    # Adjust the start and end times based on the start_time_unix
    start_time_unix += start  # Adjust start time
    end_time_unix = start_time_unix + (end - start)  # Adjust end time

    # Process the video
    video = VideoFileClip(video_path).subclip(start, end)
    video = video.resize(width=640, height=480)  # Resize the video
    video = video.fl_image(lambda x: x[..., ::-1])  # Convert color from BGR to RGB
    video.write_videofile(f'{base_path}/cut_video.mp4', fps=30, threads=1, codec="libx264")

    # Process the audio
    audio = AudioSegment.from_wav(audio_path)[start * 1000:end * 1000]
    audio.export(f'{base_path}/cut_audio.wav', format="wav")

    # Process the poses.jsonl file
    cut_poses_data = [entry for entry in poses_data if start_time_unix <= entry['timestamp'] <= end_time_unix]
    with open(f'{base_path}/cut_poses.jsonl', 'w') as cut_poses_file:
        for entry in cut_poses_data:
            cut_poses_file.write(json.dumps(entry) + '\n')

    # Process the force sensor data
    force_sensor_data = pd.read_csv(force_sensor_path)
    cut_force_sensor_data = force_sensor_data[
        (force_sensor_data['timestamp'] >= start_time_unix) & (force_sensor_data['timestamp'] <= end_time_unix)]
    cut_force_sensor_data.to_csv(f'{base_path}/cut_data.csv', index=False)

    print(f"All files in {folder} have been cut and saved with 'cut_' prefix.")
