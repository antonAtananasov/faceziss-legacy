from pulselib.processors import findFaceGetPulse
from pulselib.interface import destroyWindow, plotXY
import cv2
import numpy as np
import datetime


class PulseExtractor(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self):
        self.w, self.h = 0, 0
        self.pressed = 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(
            bpm_limits=[50, 160], data_spike_limit=2500.0, face_detector_smoothness=10.0
        )

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            self.bpm_plot = True
            self.make_bpm_plot()
            # moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY(
            [
                [self.processor.times, self.processor.samples],
                [self.processor.freqs, self.processor.fft],
            ],
            labels=[False, True],
            showmax=[False, "bpm"],
            label_ndigits=[0, 0],
            showmax_digits=[0, 1],
            skip=[3, 3],
            name=self.plot_title,
            bg=self.processor.slices[0],
        )

    def resetPlot(self):
        self.processor.data_buffer = [
            self.processor.data_buffer[-1] for _ in self.processor.data_buffer
        ]
        self.processor.times = [self.processor.times[-1] for _ in self.processor.times]

    def loop(self, frame: cv2.typing.MatLike):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        self.h, self.w, _c = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run()

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()


if __name__ == "__main__":
    App = PulseExtractor()
    while True:
        App.loop()
