# USAGE
# python facetracking.py --face cascades/haarcascade_frontalface_default.xml
# python facetracking.py --face cascades/haarcascade_frontalface_default.xml --video video/face.mov

from pyimagesearch.facedetector import FaceDetector
import argparse
import cv2

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to the face cascade XML")
ap.add_argument("-v", "--video", help="path to the optional video file")
args = vars(ap.parse_args())

# Initialize face detector
fd = FaceDetector(args["face"])

# Load video or webcam
camera = cv2.VideoCapture(args["video"]) if args.get("video", False) else cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

while True:
    (grabbed, frame) = camera.read()

    if not grabbed or frame is None:
        print("Warning: Failed to grab frame. Exiting.")
        break

    # Resize frame to width 600 while maintaining aspect ratio
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    frame = cv2.resize(frame, (600, int(600 * aspect_ratio)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameClone = frame.copy()

    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()