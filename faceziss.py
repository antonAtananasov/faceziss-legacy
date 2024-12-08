import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
from pulseDetectziss import PulseDetector
from bboxExtractor import BboxExtractor
import os
from drawUtils import putProgressBar, drawTargets
from evm import magnifyVideo, EVMModeEnum
from fileUtils import generateLineStrip, saveFrames, loadVideo
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt


DISPLAY_FPS = True
FACES_COLOR_BGR = (255, 0, 255)
HANDS_COLOR_BGR = (0, 255, 255)
PROGRESS_BAR_COLOR_BGR = (0, 255, 0)
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
CAMERA_FPS = 30
PROCESSING_WIDTH = 320
PROCESSING_HEIGHT = 240
UI_LINE_WIDTH = 1
SAMPLING_PERIOD = 3.0  # in seconds
RECORDINGS_PATH = "../Recordings/"


def showChannelIntensityPlot(
    images: list[cv2.typing.MatLike], fps: float, plotTitle: str
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    sumImage = np.sum(images, axis=0)
    if len(sumImage.shape) != 3 or sumImage.shape[2] != 3:
        raise Exception("image must contain three color chanels")
    b, g, r = (
        np.array(sumImage)[:, :, 0],
        np.array(sumImage)[:, :, 1],
        np.array(sumImage)[:, :, 2],
    )
    intensityB, intensityG, intensityR = [
        np.mean(channel, axis=0) for channel in (b, g, r)
    ]

    intensityPoints = sumImage.shape[1]
    timeline = np.linspace(0, intensityPoints / fps, intensityPoints)
    for channelIntensity, color in (
        (intensityB, "blue"),
        (intensityG, "green"),
        (intensityR, "red"),
    ):
        ax.plot(timeline, channelIntensity, color=color)

    plt.title(plotTitle)
    plt.show()


def main():
    if not os.path.isdir(RECORDINGS_PATH):
        raise FileNotFoundError(f"The directory '{RECORDINGS_PATH}' does not exist.")

    #  Window Parameters
    realWidth = WINDOW_WIDTH
    realHeight = WINDOW_HEIGHT

    # Webcam Parameters
    webcam = cv2.VideoCapture(0)

    faceDetector = FaceDetector()
    handDetector = HandDetector()

    faceBboxExtractor: BboxExtractor = BboxExtractor(SAMPLING_PERIOD, CAMERA_FPS)
    handBboxExtractor: BboxExtractor = BboxExtractor(SAMPLING_PERIOD, CAMERA_FPS)

    for bboxExtractor, name in [
        (faceBboxExtractor, "FacePulse"),
        (handBboxExtractor, "HandPulse"),
    ]:
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor: extractor.setIsRecording(False)
        )
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor: extractor.setIsRecording(False)
        )
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor: Process(
                target=showChannelIntensityPlot,
                args=[extractor.generateLineStrips(4), CAMERA_FPS, name],
            ).start()
        )

    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    prevFrameTime = time.time()
    while True:
        ret, camFrame = webcam.read()
        if ret == False:
            break

        # detect faces and hands
        _, faceBboxs = faceDetector.findFaces(camFrame, draw=False)
        handBboxs, _ = handDetector.findHands(camFrame, draw=False)

        # display fps
        uiFrame = camFrame.copy()
        if DISPLAY_FPS:
            frameTime = time.time()
            fps = 1 / (frameTime - prevFrameTime)
            prevFrameTime = frameTime
            cv2.putText(
                uiFrame,
                f"FPS: {int(fps)}",
                (30, 440),
                0,
                1,
                [0, 0, 0],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # draw targets
        drawTargets(uiFrame, faceBboxs, FACES_COLOR_BGR, UI_LINE_WIDTH)
        drawTargets(uiFrame, handBboxs, HANDS_COLOR_BGR, UI_LINE_WIDTH)

        # Face extraction
        faceBboxExtractor.loop(camFrame, faceBboxs)
        if faceBboxs and faceBboxExtractor.isRecording:
            putProgressBar(
                uiFrame,
                faceBboxs[0]["bbox"],
                faceBboxExtractor.getProgress(),
                PROGRESS_BAR_COLOR_BGR,
                UI_LINE_WIDTH,
            )
        # End Face extraction

        # Hand extraction
        handBboxExtractor.loop(camFrame, handBboxs)
        if handBboxs and handBboxExtractor.isRecording:
            putProgressBar(
                uiFrame,
                handBboxs[0]["bbox"],
                handBboxExtractor.getProgress(),
                PROGRESS_BAR_COLOR_BGR,
                UI_LINE_WIDTH,
            )
        # End Hand extraction

        # imgStack = cvzone.stackImages([uiFrame, camFrame], 2, 1)
        cv2.imshow("Faceziss", uiFrame)

        # keyboard events
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord("f"):
            faceBboxExtractor.setIsRecording(not faceBboxExtractor.isRecording)
        if keycode == ord("h"):
            handBboxExtractor.setIsRecording(not handBboxExtractor.isRecording)
        if keycode == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
