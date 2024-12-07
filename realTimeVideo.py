import cv2
from processFrame import processFrame

# USER CONFIG
WINDOW_NAME="preview"
DISPLAY_WINDOW=True

def loop(frame):
    processedFrame = processFrame(frame)

    if DISPLAY_WINDOW:
        cv2.imshow(WINDOW_NAME, processedFrame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        return False
    
    return True


def main():
   # init videocapture
    if DISPLAY_WINDOW:
        cv2.namedWindow(WINDOW_NAME)
    videocapture = cv2.VideoCapture(0)

    # videocapture settings
    # videocapture.set(cv2.CAP_PROP_FRAME_WIDTH,0)
    # videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT,0)

    while videocapture.isOpened():
        rval, frame = videocapture.read()
        if not rval: 
            break

        if not loop(frame):
            break

    
    videocapture.release()
    cv2.destroyWindow(WINDOW_NAME) 

if __name__ == "__main__":
   main()

