from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.resize(image, (900, 600))
cv2.imshow("Original", image)

chans=cv2.split(image)
colors=("b", "g","r")

plt.figure()
plt.title("’Flattened’ ColorHistogram")
plt.xlabel("Bins")
plt.ylabel("#of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)