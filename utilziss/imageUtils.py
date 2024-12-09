from enum import Enum
import numpy as np
import cv2

class LineDirectionEnum(Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"


def generateLineStrip(
    frames: list[cv2.typing.MatLike],
    lineIndexes: list[int] = [],
    lineDirection: LineDirectionEnum = LineDirectionEnum.VERTICAL,
) -> cv2.typing.MatLike:
    # (assuming all frames have the same shape) (h,w,c), c (channels)=3
    h, w, c = frames[0].shape

    if not lineIndexes:
        if lineDirection is LineDirectionEnum.VERTICAL:
            lineIndexes = [w // 2]  # center vertical line
        elif lineDirection is LineDirectionEnum.HORIZONTAL:
            lineIndexes = [h // 2]  # center horizontal line

    if lineDirection is LineDirectionEnum.VERTICAL:
        w = len(frames)  # as many columns as frames (each frame is one column)
        canvas = np.zeros((h * len(lineIndexes), w, c))
    elif lineDirection is LineDirectionEnum.HORIZONTAL:
        h = len(frames)  # as many rows as frames (each frame is one row)
        canvas = np.zeros((h, w * len(lineIndexes), c))
    else:
        raise ValueError("Unexpected line direction.")

    for i in range(len(frames)):
        frame = frames[i]
        for j in range(len(lineIndexes)):
            lineIndex = lineIndexes[j]
            if lineDirection is LineDirectionEnum.VERTICAL:
                canvas[j * h : (j + 1) * h, i] = frame[:, lineIndex]
            elif lineDirection is LineDirectionEnum.HORIZONTAL:
                canvas[i, j * w : (j + 1) * w] = frame[lineIndex]

    return canvas

def compressImages(images:list[cv2.typing.MatLike],maxPixels:int):
    # assuming all images have the same shape
    
    w,h,_ = images[0].shape

    if w*h <= maxPixels:
        return images

    else:
        newW = int(round(np.sqrt(maxPixels*w/h)))
        newH = maxPixels//newW

        return [cv2.resize(image,(newW,newH)) for image in images]