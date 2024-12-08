import time
import cv2
import numpy as np
from typing import Callable
from fileUtils import saveFrames, generateLineStrip
from evm import magnifyVideo

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
        self.isRecording:bool=False

    def setIsRecording(self, value:bool):
        self.isRecording = value

    def loop(self, frame: cv2.typing.MatLike, bboxs):
        currentFrameHasBbox = len(bboxs) == 1 and self.isRecording
        if not self._lastFrameHadBbox and currentFrameHasBbox:
            self._bboxFoundTimestamp = time.time()

        if time.time() - self._bboxFoundTimestamp >= self.requiredTime and self.isRecording:
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

    def saveFrames(self, fileName:str):            
        saveFrames(fileName, frames = self.getFrames(),fps=self.fps)

    def generateLineStrips(self, steps: int) -> list[cv2.typing.MatLike]:
        stripImages = []
        for i in range(steps):
            magnifiedVideo = magnifyVideo(
                np.array(self.getFrames()),
                self.fps,
                {
                    "freq_range": [
                        0.833 + i * 0.167 / (steps / 4),
                        1 + i * 0.167 / (steps / 4),
                    ],
                },
            )

            strip = generateLineStrip(
                magnifiedVideo,
            )

            stripImages.append(strip)
        return stripImages
