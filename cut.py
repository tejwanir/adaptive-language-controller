import json
import pandas as pd
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

folder='lab_session_1' #Change this (folder name)
start=35.5 #Change this (seconds)
end=73 #Change this (seconds)

# Define the paths to your files
poses_path = '/home/ravi/projects/adaptive-language-controller/sessions/' + folder + '/poses.jsonl'
audio_path = '/home/ravi/projects/adaptive-language-controller/sessions/' + folder + '/audio.wav'
video_path = '/home/ravi/projects/adaptive-language-controller/sessions/' + folder + '/video.avi'
force_sensor_path = '/home/ravi/projects/adaptive-language-controller/sessions/' + folder + '/data.csv'

# Read the poses.jsonl file and extract the first and last timestamps
with open(poses_path, 'r') as poses_file:
    poses_data = [json.loads(line) for line in poses_file]
    start_time_unix = poses_data[0]['timestamp']

# Define the start_time (in milliseconds) from which you want to keep the recording
# Convert the unix timestamp time 't' to milliseconds
start_time_unix = start_time_unix + start  # replace with your start_time in unix time
end_time_unix = start_time_unix + end  # replace with your end_time in unix time

# Cut the video file and resize it
video = VideoFileClip(video_path)
video = video.subclip(start, end)

# Resize the video to make it smaller (you can adjust width and height)
video = video.resize(width=640, height=480)

# Convert color from BGR to RGB
video = video.fl_image(lambda x: x[..., ::-1])

# Lower the bitrate to reduce file size (adjust as needed)
video.write_videofile('sessions/'+folder+'/cut_video.avi', fps=30, threads=1, codec="rawvideo")

# Cut the audio file
audio = AudioSegment.from_wav(audio_path)
audio = audio[start * 1000:end * 1000]
audio.export('sessions/' + folder + '/cut_audio.wav', format="wav")

# Cut the poses.jsonl file
with open(poses_path, 'r') as poses_file:
    poses_data = [json.loads(line) for line in poses_file]

cut_poses_data = [entry for entry in poses_data if start_time_unix <= entry['timestamp'] <= end_time_unix]

with open('sessions/' + folder + '/cut_poses.jsonl', 'w') as cut_poses_file:
    for entry in cut_poses_data:
        cut_poses_file.write(json.dumps(entry) + '\n')

# Cut the force sensor data
force_sensor_data = pd.read_csv(force_sensor_path)
cut_force_sensor_data = force_sensor_data[
    (force_sensor_data['timestamp'] >= start_time_unix) & (force_sensor_data['timestamp'] <= end_time_unix)]
cut_force_sensor_data.to_csv('sessions/' + folder + '/cut_data.csv', index=False)

print("All files have been cut and saved with 'cut_' prefix.")
