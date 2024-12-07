import cv2

def putBbox(
    frame: cv2.typing.MatLike,
    rect: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    linewidth,
):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, thickness=linewidth)


def drawTargets(frame, bboxs: list[dict], color: tuple[int, int, int], lineWidth=1):
    for bbox in bboxs:
        # put border around every face
        x, y, w, h = bbox["bbox"]
        putBbox(frame, bbox["bbox"], color, lineWidth)

        # for face bboxs
        if all([key in bbox.keys() for key in ["score", "id"]]):
            scorePercent = round(bbox["score"][0] * 100, 1)
            cv2.putText(
                frame,
                f"{bbox['id']}: {scorePercent}%",
                (x + 5, y + 15),
                0,
                lineWidth / 2,
                color,
                thickness=lineWidth,
                lineType=cv2.LINE_AA,
            )

        # for hand bboxs
        if "type" in bbox.keys():
            cv2.putText(
                frame,
                f"{bbox['type']}",
                (x + 5, y + 15),
                0,
                lineWidth / 2,
                color,
                thickness=lineWidth,
                lineType=cv2.LINE_AA,
            )


def putProgressBar(
    frame: cv2.typing.MatLike,
    bbox: tuple[tuple[int, int], tuple[int, int]],
    value: float,
    color_bgr: tuple[int, int, int],
    linewidth: int = 1,
):
    x, y, w, h = bbox
    borderPt1, borderPt2 = (x, y + h - 3 * linewidth), (
        x + w,
        y + h - linewidth,
    )
    cv2.rectangle(frame, borderPt1, borderPt2, color_bgr, thickness=linewidth)

    innerPt1, innerPt2 = (
        x,
        y + h - 2 * linewidth,
    ), (max(int(x + w * value), 1), y + h - 2 * linewidth)
    cv2.rectangle(
        frame,
        innerPt1,
        innerPt2,
        color_bgr,
        thickness=linewidth,
    )