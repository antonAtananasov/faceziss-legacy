import cv2
from enum import Enum
import numpy as np
import numpy.typing as npt


def saveFrames(fileName: str, frames: list[cv2.typing.MatLike] = None, fps: int = 30):
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = list(frames[0].shape[:2])
        size.reverse()
        size = tuple(size)
        videoWriter = cv2.VideoWriter(fileName, fourcc, fps, size)

        for i in range(len(frames)):
            croppedFrame = frames[i]
            videoWriter.write(croppedFrame)

        videoWriter.release()

        print(f"Saved {fileName}")

    except Exception as e:
        print("Could not save video: " + str(e))




def loadVideo(video_path:str) -> tuple[npt.NDArray, int]:
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

    video.release()

    return np.asarray(image_sequence), fps


