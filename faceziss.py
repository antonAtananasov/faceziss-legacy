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
from drawUtils import putProgressBar, drawTargets, putHints
from evm import magnifyVideo, EVMModeEnum
from fileUtils import generateLineStrip, saveFrames, loadVideo
from multiprocessing import Process, Manager
from plotUtils import plotChannelFFT, plotChannelIntensity



def main():
    DISPLAY_FPS = True
    DISPLAY_HINTS = True
    FACES_COLOR_BGR = (255, 0, 255)
    HANDS_COLOR_BGR = (0, 255, 255)
    PROGRESS_BAR_COLOR_BGR = (0, 255, 0)
    WINDOW_WIDTH = 640  # in pixels
    WINDOW_HEIGHT = 480  # in pixels
    CAMERA_FPS = 30 
    UI_LINE_WIDTH = 1 # in pixels
    SAMPLING_PERIOD = 3.0  # in seconds
    RECORDINGS_PATH = "../Recordings/"

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
            lambda extractor, name=name: Process(
                target=plotChannelIntensity,
                args=[extractor.generateLineStrips(8), CAMERA_FPS, name],
            ).start()
        )
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor, name=name: Process(
                target=plotChannelFFT,
                args=[extractor.generateLineStrips(8), CAMERA_FPS, name],
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
                (5 * UI_LINE_WIDTH, WINDOW_HEIGHT - 15 * UI_LINE_WIDTH),
                0,
                UI_LINE_WIDTH / 2,
                [0, 0, 0],
                thickness=UI_LINE_WIDTH,
                lineType=cv2.LINE_AA,
            )

        if DISPLAY_HINTS:
            putHints(uiFrame, UI_LINE_WIDTH)
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
        if keycode == ord("i"):
            DISPLAY_HINTS = not DISPLAY_HINTS
        if keycode == ord("f") or keycode == ord("b"):
            faceBboxExtractor.setIsRecording(not faceBboxExtractor.isRecording)
        if keycode == ord("h") or keycode == ord("b"):
            handBboxExtractor.setIsRecording(not handBboxExtractor.isRecording)
        if keycode == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
