import numpy as np
import cv2


def processFrame(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:

    newImg = frame.copy()
    for rect in findFacesFast(frame):
        newImg = putRect(newImg, rect, (255,128,0))

    return newImg


def findFacesFast(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faceRects = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    
    return faceRects


def generateCenterColumnImage(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    columnCount = frame.shape[1]
    centerColumnIndex = round(columnCount / 2)
    centerColumn = np.array([frame[:, centerColumnIndex]])

    centerColumnImage = centerColumn.repeat(columnCount, axis=0).transpose(1, 0, 2)
    return centerColumnImage


def putRect(img: cv2.typing.MatLike, rect,color:tuple[int,int,int]):
    x, y, w, h = rect
    r,g,b=color
    return cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (b,g,r), 4)
