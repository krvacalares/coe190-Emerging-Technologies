import cv2
import numpy as np
import argparse

# Parse command-line argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(ap.parse_args())

# Load image from argument
image = cv2.imread(args["image"])
if image is None:
    print("❌ Error: Could not load image. Check the file path.")
    exit()

# Resize to 400x400 canvas
image = cv2.resize(image, (400, 400))

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
edged = cv2.Canny(blurred, 30, 150)

# Find contours
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("I count {} coins in this image".format(len(cnts)))

# Draw contours
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 1)

# Display result
cv2.imshow('Coins', coins)
cv2.waitKey(0)