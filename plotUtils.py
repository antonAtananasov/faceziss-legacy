import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cv2
from enum import Enum


def normalizeDataArray(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculateChannelIntensity(
    lineStripLayers: list[cv2.typing.MatLike],
):
    # images should be an array of strips; and only vertical
    layerSum = np.sum(lineStripLayers, axis=0)
    if len(layerSum.shape) != 3 or layerSum.shape[2] != 3:
        raise Exception("image must contain three color chanels")
    b, g, r = (
        np.array(layerSum)[:, :, 0],
        np.array(layerSum)[:, :, 1],
        np.array(layerSum)[:, :, 2],
    )
    intensityB, intensityG, intensityR = [
        np.mean(channel, axis=0) for channel in (b, g, r)
    ]

    return intensityB, intensityG, intensityR


def calculateChannelFrequencies(
    intensityB: npt.NDArray,
    intensityG: npt.NDArray,
    intensityR: npt.NDArray,
    fps: float,
):
    ffts = []
    for channelIntensity in (intensityB, intensityG, intensityR):
        # Compute the FFT
        fft_values = np.fft.fft(channelIntensity)
        fft_freq = np.fft.fftfreq(len(channelIntensity), d=1 / fps)  # Frequencies in Hz

        # Take only the positive frequencies
        positive_freqs = fft_freq[fft_freq >= 0]
        positive_fft_values = fft_values[fft_freq >= 0]

        # Magnitude of the FFT
        magnitudes = np.abs(positive_fft_values)

        ffts.append((positive_freqs, magnitudes))  # magnitude over freq

    fftB, fftG, fftR = ffts
    return fftB, fftG, fftR


def plotChannelIntensity(
    intensityB: npt.NDArray,
    intensityG: npt.NDArray,
    intensityR: npt.NDArray,
    fps: float,
    plotTitle: str,
):
    # images should be an array of strips; and only vertical
    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    intensityPoints = intensityB.size
    timeline = np.linspace(0, intensityPoints / fps, intensityPoints)
    for channelIntensity, color in (
        (intensityB, "blue"),
        (intensityG, "green"),
        (intensityR, "red"),
    ):
        ax.plot(timeline, channelIntensity, color=color)

    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Color intensity")
    plt.title(plotTitle)
    plt.show()


class FrequenciesModeEnum(Enum):
    HZ = "Hz"
    BPM = "bpm"


def plotChannelFrequencies(
    intensityB: npt.NDArray,
    intensityG: npt.NDArray,
    intensityR: npt.NDArray,
    fps: float,
    plotTitle: str,
    frequencyMode: FrequenciesModeEnum = FrequenciesModeEnum.HZ,
):
    # images should be an array of strips; and only vertical
    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    fftB, fftG, fftR = calculateChannelFrequencies(
        intensityB, intensityG, intensityR, fps
    )

    for fft, color in (
        (fftB, "blue"),
        (fftG, "green"),
        (fftR, "red"),
    ):
        channelFreqs, freqMagnitudes = fft
        maxFreq = 2
        maxIndex = int(round(maxFreq / (np.max(channelFreqs) / channelFreqs.size))) + 1
        ax.plot(
            channelFreqs[1:maxIndex]
            * (60 if frequencyMode == FrequenciesModeEnum.BPM else 1),
            normalizeDataArray(freqMagnitudes[1:maxIndex]) * 500,
            color=color,
        )
        ax.set_xlabel(
            f"Frequency, [{'BPM'if frequencyMode == FrequenciesModeEnum.BPM else 'Hz'}]"
        )
        ax.set_ylabel("Magnitude")

    plt.title(plotTitle)
    plt.show()

def calculateCorrelation(intensitiesA:tuple[npt.NDArray,npt.NDArray,npt.NDArray],intensitiesB:tuple[npt.NDArray,npt.NDArray,npt.NDArray])->tuple[int,int,int]:
    if len(intensitiesA) !=3 or len(intensitiesB) !=3:
        raise ValueError('Intensities should have 3 channels')
    coeffs = []
    for i in range(len(intensitiesA)):
        coeff = np.corrcoef(intensitiesA[i],intensitiesB[i])[0,-1]
        coeffs.append(coeff)
    b,g,r = coeffs
    return b,g,r
