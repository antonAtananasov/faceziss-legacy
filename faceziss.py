import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

from utilziss.bboxExtractorUtils import BboxExtractor
from utilziss.drawUtils import putProgressBar, drawTargets, putHints, putFps
from evmlib.evm import magnifyVideoWithFreqSteps
from utilziss.imageUtils import generateLineStrip, compressImages
from multiprocessing import Process
from utilziss.plotUtils import (
    plotChannelIntensity,
    calculateChannelIntensity,
    calculateCorrelation,
    calculateCorrelationMatch,
)
from utilziss.fileUtils import saveFrames
from pulselib.getPulse import PulseExtractor
from utilziss.faceEncodingUtils import (
    generateFaceEncoding,
    saveSubjectsFaceData,
    loadSubjectsFaceData,
    compareFaces,
    generateRandomCancellableTemplate,
)
from utilziss.encryptUtils import Encryptor

USAGE_HINTS = (
    "Q - Close window",
    "F - Get pulse wave from face",
    "H - Get pulse wave from hand",
    "B - Get pulse wave from both",
    "P - Get pulse wave from finger",
    "E - Get face encoding",
    "R - Register face encoding",
    "I - Show/hide hints",
)


def main():
    DISPLAY_FPS = True
    DISPLAY_HINTS = True
    FACES_COLOR_BGR = (255, 0, 255)
    HANDS_COLOR_BGR = (0, 255, 255)
    PROGRESS_BAR_COLOR_BGR = (0, 255, 0)
    UI_TEXT_COLOR = (0, 0, 0)
    DANGER_COLOR_BGR = (0, 0, 255)
    SUCCESS_COLOR_BGR = (0, 255, 0)
    WARNING_COLOR_BGR = (0, 135, 252)
    RED_MATCH_THRESHOLD, GREEN_MATCH_THRESHOLD, BLUE_MATCH_THRESHOLD = (
        0.7,
        0.0,
        0.6,
    )  # correlation coefficients for succesfull face-to-hand match
    FACE_MATCH_TOLERANCE = 0.6  # face biometric match tolerance = 1 - threshold
    WINDOW_WIDTH = 1280  # in pixels
    WINDOW_HEIGHT = 720  # in pixels
    PROCESSING_MAX_PIXELS = (
        80 * 80
    )  # maximum amount of pixels to be used to rescale samples for magnification
    CAMERA_FPS = 30
    UI_LINE_WIDTH = 1  # in pixels
    SAMPLING_PERIOD = 3.0  # in seconds
    ENCODINGS_PATH = "./encodings/"
    RECORDINGS_PATH = "../Recordings/"
    RECORD_OUTPUTS = True  # whether to export data used to calculate stuff

    if not os.path.isdir(RECORDINGS_PATH):
        raise FileNotFoundError(f"The directory '{RECORDINGS_PATH}' does not exist.")

    #  Window Parameters
    realWidth = WINDOW_WIDTH
    realHeight = WINDOW_HEIGHT

    # Webcam Parameters
    webcam = cv2.VideoCapture(0)
    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    faceDetector = FaceDetector()
    handDetector = HandDetector()

    faceBboxExtractor: BboxExtractor = BboxExtractor(SAMPLING_PERIOD, CAMERA_FPS)
    handBboxExtractor: BboxExtractor = BboxExtractor(SAMPLING_PERIOD, CAMERA_FPS)

    recordedIntensities = {}
    recordedIntensitiesCorrelate = False

    faceIdentificatorName = "FacePulse"
    handIdentificatorName = "HandPulse"
    for bboxExtractor, name in [
        (faceBboxExtractor, faceIdentificatorName),
        (handBboxExtractor, handIdentificatorName),
    ]:
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor: extractor.setIsRecording(False)
        )
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor: extractor.setIsRecording(False)
        )
        bboxExtractor.onSuccessHandlers.append(
            lambda extractor, name=name: (
                filename := os.path.join(RECORDINGS_PATH, f"{name}_{time.time()}"),
                compressedImages := compressImages(
                    extractor.getFrames(), PROCESSING_MAX_PIXELS
                ),
                magnifiedVideos := magnifyVideoWithFreqSteps(
                    compressedImages,
                    extractor.fps,
                    8,
                ),
                (
                    [
                        saveFrames(
                            filename + f"_{i}.mp4",
                            magnifiedVideos[i],
                            CAMERA_FPS,
                        )
                        for i in range(len(magnifiedVideos))
                    ]
                    if RECORD_OUTPUTS
                    else None
                ),
                channelIntensities := calculateChannelIntensity(
                    [generateLineStrip(video) for video in magnifiedVideos]
                ),
                recordedIntensities.update({name: channelIntensities}),
                (
                    np.savetxt(filename + f".csv", channelIntensities, delimiter=",")
                    if RECORD_OUTPUTS
                    else None
                ),
                (
                    [
                        cv2.imwrite(
                            filename + f"_{i}.jpg",
                            generateLineStrip(magnifiedVideos[i]),
                        )
                        for i in range(len(magnifiedVideos))
                    ]
                    if RECORD_OUTPUTS
                    else None
                ),
                Process(
                    target=plotChannelIntensity,
                    args=[
                        channelIntensities[0],
                        channelIntensities[1],
                        channelIntensities[2],
                        CAMERA_FPS,
                        name,
                    ],
                ).start(),
            )
        )

    pulseExtractor = PulseExtractor(ENCODINGS_PATH)
    pulseExtractor.make_bpm_plot()

    knownFaceEncodings: dict[str, list[float]] = loadSubjectsFaceData(
        os.path.join(ENCODINGS_PATH, "knownFaceEncodings.enc")
    )
    faceObfuscators: dict[str, list[float]] = loadSubjectsFaceData(
        os.path.join(ENCODINGS_PATH, "knownFaceObfuscators.enc")
    )
    successfullMatchingFaces = []

    encryptor = Encryptor()

    prevFrameTime = time.time()
    while True:
        ret, camFrame = webcam.read()
        if ret == False:
            break

        # Display FPS
        uiFrame = camFrame.copy()
        if DISPLAY_FPS:
            putFps(uiFrame, prevFrameTime, UI_TEXT_COLOR, UI_LINE_WIDTH)
        prevFrameTime = time.time()

        if DISPLAY_HINTS:
            putHints(uiFrame, USAGE_HINTS, UI_LINE_WIDTH)

        # Detect faces and hands
        _, faceBboxs = faceDetector.findFaces(camFrame, draw=False)
        handBboxs, _ = handDetector.findHands(camFrame, draw=False)
        # Draw All BBoxs
        drawTargets(uiFrame, faceBboxs, FACES_COLOR_BGR, UI_LINE_WIDTH)
        drawTargets(uiFrame, handBboxs, HANDS_COLOR_BGR, UI_LINE_WIDTH)

        # Begin Face extraction
        faceBboxExtractor.loop(camFrame, faceBboxs)
        if faceBboxs and faceBboxExtractor.isRecording:
            putProgressBar(
                uiFrame,
                faceBboxs[0]["bbox"],
                faceBboxExtractor.getProgress(),
                PROGRESS_BAR_COLOR_BGR,
                UI_LINE_WIDTH,
            )
        if len(faceBboxs) != 1:
            faceBboxExtractor.setIsRecording(False)
        # End Face extraction

        # Begin Hand extraction
        handBboxExtractor.loop(camFrame, handBboxs)
        if handBboxs and handBboxExtractor.isRecording:
            putProgressBar(
                uiFrame,
                handBboxs[0]["bbox"],
                handBboxExtractor.getProgress(),
                PROGRESS_BAR_COLOR_BGR,
                UI_LINE_WIDTH,
            )
        if len(handBboxs) != 1:
            handBboxExtractor.setIsRecording(False)

        # End Hand extraction

        # Begin Correlation
        if (
            faceIdentificatorName in recordedIntensities.keys()
            and handIdentificatorName in recordedIntensities.keys()
        ) and (
            len(recordedIntensities[faceIdentificatorName][0])
            == len(recordedIntensities[handIdentificatorName][0])
        ):
            channelCorrelCoefs = calculateCorrelation(
                recordedIntensities[faceIdentificatorName],
                recordedIntensities[handIdentificatorName],
            )
            recordedIntensitiesCorrelate = calculateCorrelationMatch(
                (RED_MATCH_THRESHOLD, GREEN_MATCH_THRESHOLD, BLUE_MATCH_THRESHOLD),
                channelCorrelCoefs,
                True,
            )
            recordedIntensities.clear()

        correlationMatchText = (
            f"Hand-Face Match: {'Yes' if recordedIntensitiesCorrelate else 'No'}"
        )
        cv2.putText(
            uiFrame,
            correlationMatchText,
            (uiFrame.shape[1] - len(correlationMatchText) * 10, uiFrame.shape[0] - 15),
            0,
            UI_LINE_WIDTH / 2,
            (
                WARNING_COLOR_BGR
                if faceBboxExtractor.isRecording or handBboxExtractor.isRecording
                else (
                    SUCCESS_COLOR_BGR
                    if recordedIntensitiesCorrelate
                    else DANGER_COLOR_BGR
                )
            ),
            thickness=UI_LINE_WIDTH,
            lineType=cv2.LINE_AA,
        )
        # End Correlation

        # Begin Finger pulse extraction
        pulseExtractor.loop(
            camFrame[
                camFrame.shape[0] // 2 - 20 : camFrame.shape[0] // 2 + 20,
                camFrame.shape[1] // 2 - 20 : camFrame.shape[1] // 2 + 20,
            ]
        )
        # End Finger pulse extraction
        recognitionMatchText = f"Matched face: {','.join(successfullMatchingFaces) if successfullMatchingFaces else 'None'}"
        cv2.putText(
            uiFrame,
            recognitionMatchText,
            (
                uiFrame.shape[1] - (len(recognitionMatchText) - 1) * 10,
                uiFrame.shape[0] - 15 * 2,
            ),
            0,
            UI_LINE_WIDTH / 2,
            (SUCCESS_COLOR_BGR if successfullMatchingFaces else DANGER_COLOR_BGR),
            thickness=UI_LINE_WIDTH,
            lineType=cv2.LINE_AA,
        )

        # Begin Face Recognition

        # End Face Recognition

        cv2.imshow("Faceziss", uiFrame)

        # Begin Keyboard events
        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord("i"):
            DISPLAY_HINTS = not DISPLAY_HINTS

        if keycode == ord("p"):
            pulseExtractor.toggle_display_plot()
            if not pulseExtractor.bpm_plot and RECORD_OUTPUTS:
                for values, name in (
                    (pulseExtractor.processor.times, "times"),
                    (pulseExtractor.processor.samples, "samples"),
                    (pulseExtractor.processor.freqs, "freqs"),
                    (pulseExtractor.processor.fft, "fft"),
                ):
                    np.savetxt(
                        os.path.join(
                            RECORDINGS_PATH, f"finger_{time.time()}_{name}.csv"
                        ),
                        values,
                        delimiter=",",
                    )
            pulseExtractor.resetPlot()

        if keycode == ord("f"):
            if len(faceBboxs) == 1:
                faceBboxExtractor.setIsRecording(not faceBboxExtractor.isRecording)
            else:
                print("Target not on screen or too many targets!")

        if keycode == ord("h"):
            if len(handBboxs) == 1:
                handBboxExtractor.setIsRecording(not handBboxExtractor.isRecording)
            else:
                print("Target not on screen or too many targets!")

        if keycode == ord("b"):
            recordedIntensities.clear()
            if len(faceBboxs) == 1 and len(handBboxs) == 1:
                faceBboxExtractor.setIsRecording(not faceBboxExtractor.isRecording)
                handBboxExtractor.setIsRecording(not handBboxExtractor.isRecording)
            else:
                print("Targets not on screen or too many targets!")

        if keycode == ord("e") or keycode == ord("r"):
            if len(faceBboxs) == 1:
                x, y, w, h = faceBboxs[0]["bbox"]
                faceFrame = camFrame[
                    np.clip(y, 0, camFrame.shape[0]) : np.clip(
                        y + h, 0, camFrame.shape[0]
                    ),
                    np.clip(x, 0, camFrame.shape[1]) : np.clip(
                        x + w, 0, camFrame.shape[1]
                    ),
                ]
                compressedFaceFrame = compressImages(
                    [faceFrame], PROCESSING_MAX_PIXELS * 2
                )[0]
                _, faceEncodings = generateFaceEncoding(compressedFaceFrame)
                if faceEncodings:
                    faceEncoding = faceEncodings[0]
                    if RECORD_OUTPUTS:
                        np.savetxt(
                            os.path.join(
                                RECORDINGS_PATH, f"faceCode_{time.time()}.csv"
                            ),
                            faceEncoding,
                            delimiter=",",
                        )
                        binaryEncodingsPath = os.path.join(
                            RECORDINGS_PATH, f"faceCode_{time.time()}.bin"
                        )
                        saveSubjectsFaceData(
                            binaryEncodingsPath, {"CurrentUser": [faceEncoding]}
                        )

                    comparisons = compareFaces(
                        knownFaceEncodings,
                        faceObfuscators,
                        faceEncoding,
                        FACE_MATCH_TOLERANCE,
                    )
                    print(comparisons)
                    acceptedSubjects = [
                        subject
                        for subject, comparisonValues in comparisons.items()
                        if any(comparisonValues)
                    ]
                    successfullMatchingFaces = acceptedSubjects

                    if keycode == ord("r"):
                        obfuscation, cancellableEncoding = (
                            generateRandomCancellableTemplate(faceEncoding)
                        )
                        subjectName = input("Enter Subject Name: ")
                        if subjectName:
                            if subjectName in knownFaceEncodings:
                                knownFaceEncodings[subjectName].append(
                                    cancellableEncoding
                                )
                            else:
                                knownFaceEncodings[subjectName] = [cancellableEncoding]
                            faceObfuscators[subjectName] = obfuscation
                        saveSubjectsFaceData(
                            os.path.join(ENCODINGS_PATH, "knownFaceEncodings.enc"),
                            knownFaceEncodings,
                        )
                        saveSubjectsFaceData(
                            os.path.join(ENCODINGS_PATH, "knownFaceObfuscators.enc"),
                            faceObfuscators,
                        )
                        if RECORD_OUTPUTS:
                            np.savetxt(
                                os.path.join(
                                    RECORDINGS_PATH,
                                    f"{subjectName}_faceCancellableEncoding_{time.time()}.csv",
                                ),
                                cancellableEncoding,
                                delimiter=",",
                            )
                            np.savetxt(
                                os.path.join(
                                    RECORDINGS_PATH,
                                    f"{subjectName}_faceObfuscator_{time.time()}.csv",
                                ),
                                obfuscation,
                                delimiter=",",
                            )

                else:
                    print("Unable to track face!")
            else:
                print("Target not on screen or too many targets!")

        if keycode == ord("q"):
            break
        # End Keyboard events

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Welcome to Faceziss!")
    for hint in USAGE_HINTS:
        print(hint)
    print("Follow the text in the window for more information.")
    print()
    main()
    print("Program exit.")
