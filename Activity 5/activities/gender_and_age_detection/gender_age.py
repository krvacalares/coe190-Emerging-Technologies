import os
import cv2
import argparse

# Define constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceProto = "saved_models/opencv_face_detector.pbtxt"
faceModel = "saved_models/opencv_face_detector_uint8.pb"
ageProto = "saved_models/age_deploy.prototxt"
ageModel = "saved_models/age_net.caffemodel"
genderProto = "saved_models/gender_deploy.prototxt"
genderModel = "saved_models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=True)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameOpencvDnn, faceBoxes

def inference(frame, faceBox, padding=20):
    face = frame[
        max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
        max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)
    ]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age, gender

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())

    video = args["video"] if args["video"] is not None else 0
    cap = cv2.VideoCapture(video)

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to grab frame. Exiting.")
            break

        resultFrame, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            age, gender = inference(frame, faceBox)
            label = f"{gender}, {age}"
            cv2.putText(resultFrame, label, (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age and Gender Detection", resultFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()