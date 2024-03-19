import json
import multiprocessing as mp
import socket
import time
from typing import Optional, TypedDict

import cv2
import numpy as np

from filter_poses import FrameCorrector, PoseFilter, UserPoses, preprocess_frame


class OpenSensorRequest(TypedDict):
    type: str
    requested_index: int
    requested_color_width: int
    requested_color_height: int
    requested_depth_width: int
    requested_depth_height: int
    requested_fps: int
    smoothing: float


COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480


def collect_poses(
    output: "Optional[mp.Queue[tuple[float, UserPoses]]]",
):
    depth_filter = PoseFilter()
    frame_corrector = FrameCorrector(0)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", 60001))
        req = OpenSensorRequest(
            {
                "type": "RealSense",
                "requested_index": 0,
                "requested_color_width": COLOR_WIDTH,
                "requested_color_height": COLOR_HEIGHT,
                "requested_depth_width": DEPTH_WIDTH,
                "requested_depth_height": DEPTH_HEIGHT,
                "requested_fps": 30,
                "smoothing": 0.1,
            }
        )
        json_req = json.dumps(req) + "\n"
        s.sendall(json_req.encode("utf-8"))

        buf_size = 1024
        buf: bytes

        # Receive the reply
        buf = s.recv(buf_size)
        reply_buf = b""
        while True:
            try:
                idx = buf.index(b"\n")
                reply_buf += buf[:idx]
                buf = buf[idx + 1 :]
                break
            except ValueError:
                reply_buf += buf
                buf = s.recv(buf_size)
        reply = json.loads(reply_buf.decode("utf-8"))
        color_format = reply["color_format"]

        # Receive the frames
        # Increase the buffer size to 8KB
        buf_size = 8192
        while True:
            # Read color frame
            while len(buf) < 8:
                buf += s.recv(buf_size)
            color_data_size = int.from_bytes(buf[:8], "little")
            buf = buf[8:]
            while len(buf) < color_data_size:
                buf += s.recv(buf_size)
            color_data = buf[:color_data_size]
            buf = buf[color_data_size:]
            # Read depth frame
            while len(buf) < 8:
                buf += s.recv(buf_size)
            depth_data_size = int.from_bytes(buf[:8], "little")
            buf = buf[8:]
            while len(buf) < depth_data_size:
                buf += s.recv(buf_size)
            depth_data = buf[:depth_data_size]
            buf = buf[depth_data_size:]
            json_buf = b""
            while True:
                try:
                    idx = buf.index(b"\n")
                    json_buf += buf[:idx]
                    buf = buf[idx + 1 :]
                    break
                except ValueError:
                    json_buf += buf
                    buf = s.recv(buf_size)
            frame = json.loads(json_buf.decode("utf-8"))

            # From bytes to numpy arrays
            color_data = np.frombuffer(color_data, dtype=np.uint8)
            depth_data = np.frombuffer(depth_data, dtype=np.uint16)
            # Convert color_data to a format that can be displayed
            if color_format == "RGB":
                color_data = color_data.reshape((COLOR_HEIGHT, COLOR_WIDTH, 3))
                color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
            elif color_format == "ARGB":
                color_data = color_data.reshape((COLOR_HEIGHT, COLOR_WIDTH, 4))
                color_data = cv2.cvtColor(color_data[:, :, 1:], cv2.COLOR_RGB2BGR)
            elif color_format == "Mono":
                color_data = color_data.reshape((COLOR_HEIGHT, COLOR_WIDTH))
            else:
                raise ValueError(f"Unknown color format: {color_format}")
            # Annotate joints on the color frame
            for body in frame["skeletons"]:
                for joint_name, joint in body.items():
                    if joint_name == "user_id" or joint_name == "confidence":
                        continue
                    if joint["confidence"] < 0.5:
                        continue
                    pos_2d = np.asarray(joint["pos2D"])
                    pos_2d = (int(pos_2d[0]), int(pos_2d[1]))
                    cv2.circle(color_data, pos_2d, 5, (0, 255, 0), -1)
                    cv2.putText(
                        color_data,
                        joint_name,
                        pos_2d,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            cv2.imshow("frame", color_data)

            if output is not None:
                timestamp, poses = preprocess_frame(frame, 0.5)
                timestamp /= 1e9
                poses = depth_filter.filter_skeletons(timestamp, poses)
                poses = frame_corrector.filter_skeletons(poses)
                output.put((timestamp, poses))

            # Quit on esc
            if cv2.waitKey(1) == 27:
                break

            time.sleep(0.001)


if __name__ == "__main__":
    collect_poses()
