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


class LineDirectionEnum(Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"


def generateLineStrip(
    frames: list[cv2.typing.MatLike],
    lineIndexes: list[int] = [],
    lineDirection: LineDirectionEnum = LineDirectionEnum.VERTICAL,
) -> cv2.typing.MatLike:
    # (assuming all frames have the same shape) (h,w,c), c (channels)=3
    h, w, c = frames[0].shape

    if not lineIndexes:
        if lineDirection is LineDirectionEnum.VERTICAL:
            lineIndexes = [w // 2]  # center vertical line
        elif lineDirection is LineDirectionEnum.HORIZONTAL:
            lineIndexes = [h // 2]  # center horizontal line

    if lineDirection is LineDirectionEnum.VERTICAL:
        w = len(frames)  # as many columns as frames (each frame is one column)
        canvas = np.zeros((h*len(lineIndexes), w, c))
    elif lineDirection is LineDirectionEnum.HORIZONTAL:
        h = len(frames)  # as many rows as frames (each frame is one row)
        canvas = np.zeros((h, w*len(lineIndexes), c))
    else:
        raise ValueError("Unexpected line direction.")


    for i in range(len(frames)):
        frame = frames[i]
        for j in range(len(lineIndexes)):
            lineIndex = lineIndexes[j]
            if lineDirection is LineDirectionEnum.VERTICAL:
                canvas[j * h : (j + 1) * h, i] = frame[:, lineIndex]
            elif lineDirection is LineDirectionEnum.HORIZONTAL:
                canvas[i, j * w : (j + 1) * w] = frame[lineIndex]

    return canvas


def loadVideo(video_path) -> tuple[npt.NDArray, int]:
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
