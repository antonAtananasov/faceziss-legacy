import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalizeDataArray(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plotChannelIntensity(
    lineStripLayers: list[cv2.typing.MatLike], fps: float, plotTitle: str
):
    # images should be an array of strips; and only vertical
    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    sumImage, (intensityB, intensityG, intensityR), _ = (
        calculateChannelIntensity(lineStripLayers, fps)
    )

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

def plotChannelFFT(
    lineStripLayers: list[cv2.typing.MatLike], fps: float, plotTitle: str
):
    # images should be an array of strips; and only vertical
    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    _, _, (fftB, fftG, fftR) = (
        calculateChannelIntensity(lineStripLayers, fps)
    )

    for fft, color in (
        (fftB, "blue"),
        (fftG, "green"),
        (fftR, "red"),
    ):
        channelFreqs, freqMagnitudes = fft
        maxFreq = 2
        maxIndex = int(round(maxFreq/(np.max(channelFreqs)/channelFreqs.size)))+1
        ax.plot(channelFreqs[1:maxIndex],normalizeDataArray(freqMagnitudes[1:maxIndex])*500, color=color)

    plt.title(plotTitle)
    plt.show()
    


def calculateChannelIntensity(lineStripLayers: list[cv2.typing.MatLike], fps: float):
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

    return layerSum, (intensityB, intensityG, intensityR), (fftB, fftG, fftR)
