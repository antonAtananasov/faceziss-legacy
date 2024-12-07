import numpy as np
import cv2


class PulseDetector:

    def __init__(self):
        self.realWidth = 640
        self.realHeight = 480
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 15

        # Color Magnification Parameters
        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIndex = 0

        # Initialize Gaussian Pyramid
        self.firstFrame = np.zeros(
            (self.videoHeight, self.videoWidth, self.videoChannels)
        )
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels + 1)[self.levels]
        self.videoGauss = np.zeros(
            (
                self.bufferSize,
                self.firstGauss.shape[0],
                self.firstGauss.shape[1],
                self.videoChannels,
            )
        )
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        # Bandpass Filter for Specified Frequencies
        self.frequencies = (
            (1.0 * self.videoFrameRate)
            * np.arange(self.bufferSize)
            / (1.0 * self.bufferSize)
        )
        self.mask = (self.frequencies >= self.minFrequency) & (
            self.frequencies <= self.maxFrequency
        )

        # Heart Rate Calculation Variables
        self.bpmCalculationFrequency = 10  # 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))
        self.i = 0
        self.ptime = 0
        self.ftime = 0

        self.BPM = -1

    # Helper Methods
    def buildGauss(self, frame: cv2.typing.MatLike, levels: int):
        pyramid = [frame]
        for _ in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(self, pyramid, index: int, levels: int):
        filteredFrame = pyramid[index]
        for _ in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[: self.videoHeight, : self.videoWidth]
        return filteredFrame

    def detection_loop(self, bboxFrame: cv2.typing.MatLike):
        # one iteration of the detection loop

        try:
            detectionFrame = bboxFrame
            detectionFrame = cv2.resize(
                detectionFrame, (self.videoWidth, self.videoHeight)
            )

            # Construct Gaussian Pyramid
            self.videoGauss[self.bufferIndex] = self.buildGauss(
                detectionFrame, self.levels + 1
            )[self.levels]
            fourierTransform = np.fft.fft(self.videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[self.mask == False] = 0

            # Grab a Pulse
            if self.bufferIndex % self.bpmCalculationFrequency == 0:
                self.i = self.i + 1
                for buf in range(self.bufferSize):
                    self.fourierTransformAvg[buf] = np.real(
                        fourierTransform[buf]
                    ).mean()
                hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
                bpm = 60.0 * hz
                self.bpmBuffer[self.bpmBufferIndex] = bpm
                self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize

            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * self.alpha

            # Reconstruct Resulting Frame
            filteredFrame = self.reconstructFrame(
                filtered, self.bufferIndex, self.levels
            )
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize
            outputFrame_show = cv2.resize(
                outputFrame, (self.videoWidth // 2, self.videoHeight // 2)
            )

            bpm_value = self.bpmBuffer.mean()

            if self.i > self.bpmBufferSize:
                self.BPM = bpm_value
            else:
                self.BPM = -1

            return outputFrame, outputFrame_show
        except:
            pass
