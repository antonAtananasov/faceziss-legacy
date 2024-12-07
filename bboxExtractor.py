import time
import cv2
import numpy as np
from typing import Callable


class BboxExtractor:
    def __init__(self, requiredTime: float = 3, fps:int=30):
        self._bboxFoundTimestamp = (
            time.time()
        )  # first moment when face appears in the frame
        self.savedFrames: list[cv2.typing.MatLike] = []  # holds collected frames
        self.savedBboxs: list[dict] = []  # holds collected frames
        self.requiredTime: float = (
            requiredTime  # continuous time in seconds to collect frames
        )
        self.fps:int=fps
        self._lastFrameHadBbox = False
        self.onSuccessHandlers: list[Callable[[BboxExtractor],None]] = []

    def loop(self, frame: cv2.typing.MatLike, bboxs):
        currentFrameHasBbox = len(bboxs) == 1
        if not self._lastFrameHadBbox and currentFrameHasBbox:
            self._bboxFoundTimestamp = time.time()

        if time.time() - self._bboxFoundTimestamp >= self.requiredTime:
            for eventHandler in self.onSuccessHandlers:
                eventHandler(self)

                # reset
            self._bboxFoundTimestamp = time.time()
            self.savedFrames.clear()

        self._lastFrameHadBbox = currentFrameHasBbox

        if currentFrameHasBbox:
            x, y, w, h = bboxs[0]["bbox"]
            self.savedFrames.append(frame)
            self.savedBboxs.append(bboxs[0])
        else:
            # reset
            self.savedFrames.clear()
            self.savedBboxs.clear()
            self._bboxFoundTimestamp = time.time()

    def getProgress(self):
        return np.clip(
            (time.time() - self._bboxFoundTimestamp) / self.requiredTime, 0, 1
        )

    def getFrames(
        self, cropToBbox: bool = True, size: tuple[int, int] = (-1, -1)
    ) -> list[cv2.typing.MatLike]:
        frames = self.savedFrames.copy()
        if cropToBbox:
            # ensure size
            minCoords = np.array(
                [savedBbox["bbox"][:2] for savedBbox in self.savedBboxs]
            )
            maxCoords = np.array(
                [
                    np.array(savedBbox["bbox"][:2]) + np.array(savedBbox["bbox"][2:])
                    for savedBbox in self.savedBboxs
                ]
            )
            # Compute min and max for x and y
            minX, maxX = minCoords[:, 0].min(), maxCoords[:, 0].max()
            minY, maxY = minCoords[:, 1].min(), maxCoords[:, 1].max()
            clippedMinY, clippedMaxY = np.clip(minY, 0, frames[0].shape[0]), np.clip(
                maxY, 0, frames[0].shape[0]
            )
            clippedMinX, clippedMaxX = np.clip(minX, 0, frames[0].shape[1]), np.clip(
                maxX, 0, frames[0].shape[1]
            )

            frames = [
                savedFrame[clippedMinY:clippedMaxY, clippedMinX:clippedMaxX]
                for savedFrame in self.savedFrames
            ]
        if all([length > 0 for length in size]):
            return [cv2.resize(bboxFrame, size) for bboxFrame in self.savedFrames]
        else:
            return frames

    def saveFrames(self, fileName:str,frames:list[cv2.typing.MatLike]=None):
        if frames is None:
            frames = self.getFrames()

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size =list(frames[0].shape[:2])
            size.reverse()
            size = tuple(size)
            videoWriter = cv2.VideoWriter(fileName, fourcc, self.fps, size)

            for i in range(len(frames)):
                croppedFrame = frames[i]
                videoWriter.write(croppedFrame)

            videoWriter.release()

            print(f"Saved {fileName}")

        except Exception as e:
            print("Could not save video: " + str(e))
