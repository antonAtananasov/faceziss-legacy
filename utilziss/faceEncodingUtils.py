from pathlib import Path
import face_recognition
import pickle
from typing import Dict, Any
import cv2
import numpy.typing as npt
import numpy as np
import os


def generateFaceEncoding(
    img: cv2.typing.MatLike, model="hog"
) -> tuple[list[tuple[int, Any, Any, int]], Dict]:
    face_locations = face_recognition.face_locations(img, model=model)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    return face_locations, face_encodings


def generateFaceEncodings(trainingPath: str, model="hog") -> Dict:
    subjectNames = []
    faceEncodings = []

    for filepath in Path(trainingPath).glob("*/*"):
        subjectName = filepath.parent.name
        imageFile = face_recognition.load_image_file(filepath)

        _, generatedFaceEncodings = generateFaceEncoding(imageFile, model)
        faceEncodings += generatedFaceEncodings
        subjectNames += [subjectName] * len(generatedFaceEncodings)

    name_encodings = dict(zip(subjectNames, faceEncodings))
    return name_encodings


def saveFaceEncodings(outputFilePath: str, name_encodings: Dict) -> None:
    with open(outputFilePath, mode="wb") as f:
        pickle.dump(name_encodings, f)


def loadFaceEncodings(encodingsFilePath: str) -> Dict:
    if not os.path.exists(encodingsFilePath):
        return {}
    loadedEncodings = None
    with open(encodingsFilePath, mode="rb") as f:
        loadedEncodings = pickle.load(f)
    return loadedEncodings


def compareFaces(
    knownEncodings: dict, encodingToCheck: npt.NDArray, tolerance: float = 0.6
) -> dict[str,list[bool]]:
    compareSubjects = {}
    for subject, encodings in knownEncodings.items():
        compareResults = face_recognition.compare_faces(
            encodings, encodingToCheck, tolerance
        )

        compareSubjects[subject] = compareResults
    # requires all values to be under the tolerance
    return compareSubjects
