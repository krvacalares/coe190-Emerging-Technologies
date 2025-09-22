# USAGE
# python track.py
# python track.py --video video/iphonecase.mov
# python track.py --color red

import numpy as np
import argparse
import time
import cv2

# Define color boundaries in BGR format
def color_range(color):
    if color == 'blue':
        lower = np.array([100, 67, 0], dtype="uint8")
        upper = np.array([255, 128, 50], dtype="uint8")
        return (lower, upper)
    elif color == 'green':
        lower = np.array([0, 100, 0], dtype="uint8")
        upper = np.array([100, 255, 100], dtype="uint8")
        return (lower, upper)
    elif color == 'red':
        lower = np.array([0, 0, 100], dtype="uint8")   # low blue, low green, high red
        upper = np.array([80, 80, 255], dtype="uint8")
        return (lower, upper)
    else:
        raise ValueError("Unsupported color. Choose from 'blue', 'green', or 'red'.")

# Match rectangle color to tracked color
def get_rect_color(color_name):
    if color_name == 'red':
        return (0, 0, 255)
    elif color_name == 'green':
        return (0, 255, 0)
    elif color_name == 'blue':
        return (255, 0, 0)
    else:
        return (255, 255, 255)  # default to white

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to the (optional) video file")
    ap.add_argument("-c", "--color", required=False, default="blue", help="color option is blue, green, or red (default is blue)")
    args = vars(ap.parse_args())

    video = args["video"] if args["video"] is not None else 0
    camera = cv2.VideoCapture(video)

    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        # Get color boundaries and rectangle color
        colorLower, colorUpper = color_range(args["color"])
        rect_color = get_rect_color(args["color"])

        # Create binary mask and blur it
        mask = cv2.inRange(frame, colorLower, colorUpper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Find contours
        (cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(frame, [rect], -1, rect_color, 2)

        # Show frames
        cv2.imshow("Tracking", frame)
        cv2.imshow("Binary", mask)

        time.sleep(0.025)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()