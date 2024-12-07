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
from evm import processVideo

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
SAMPLING_PERIOD = 5.0  # 3.0 in seconds
RECORDINGS_PATH = "../Recordings/"


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
    # faceBboxExtractor.onSuccessHandlers.append(
    #     lambda extractor: extractor.saveFrames(
    #         os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4")
    #     )
    # )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833 , 1]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833+.167 , 1+.167]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833+.167+.167 , 1+.167+.167]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833+.167+.167+.167 , 1+.167+.167+.167]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833+.167+.167+.167+.167 , 1+.167+.167+.167+.167]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833 +.167+.167+.167+.167+.167, 1+.167+.167+.167+.167+.167]},
            ),
        )
    )
    faceBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"face_{time.time()}.mp4"),
            processVideo(
                np.array(extractor.getFrames()),
                extractor.fps,
                {"freq_range": [0.833 +.167+.167+.167+.167+.167+.167, 1+.167+.167+.167+.167+.167+.167]},
            ),
        )
    )
    handBboxExtractor.onSuccessHandlers.append(
        lambda extractor: extractor.saveFrames(
            os.path.join(RECORDINGS_PATH, f"hand_{time.time()}.mp4")
        )
    )

    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    lastFrameHadHand = False
    handFoundTimestamp = 0
    handFrames = []

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
        if len(faceBboxs) == 1:
            # TODO: make prettier
            # TODO: fix len(facebboxs)
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
        if len(handBboxs) == 1:
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
