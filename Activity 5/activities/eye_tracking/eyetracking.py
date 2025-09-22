from pyimagesearch.eyetracker import EyeTracker
import argparse
import cv2
import os

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to the face cascade XML")
ap.add_argument("-e", "--eye", required=True, help="path to the eye cascade XML")
ap.add_argument("-v", "--video", help="path to the optional video file")
args = vars(ap.parse_args())

# Initialize EyeTracker
et = EyeTracker(args["face"], args["eye"])

# Load video or webcam
camera = cv2.VideoCapture(args["video"]) if args.get("video", False) else cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    (grabbed, frame) = camera.read()

    if not grabbed or frame is None:
        print("⚠️ Warning: Failed to grab frame. Exiting.")
        break

    # Resize frame to width 600 while maintaining aspect ratio
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    frame = cv2.resize(frame, (600, int(600 * aspect_ratio)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    tracked = et.track(gray)

    for face_rect, eye_rects in tracked:
        cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 2)
        for eye in eye_rects:
            cv2.rectangle(frame, (eye[0], eye[1]), (eye[2], eye[3]), (255, 0, 0), 2)

    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()