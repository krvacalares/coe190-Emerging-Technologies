from __future__ import print_function
from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.resize(image, (600, 900))
cv2.imshow("Original", image)

#-----------------------------------------#

quadrants = {
    "Quadrant 1 (Top-Left)": image[0:450, 0:300],
    "Quadrant 2 (Bottom-Left)": image[0:450, 300:600],
    "Quadrant 3 (Top-Right)": image[450:900, 0:300],
    "Quadrant 4 (Bottom-Right)": image[450:900, 300:600]
}

#-----------------------------------------#

def price_calculator(corner, title):
    
    chans = cv2.split(corner)
    white_pixels = 0

    plt.figure()
    plt.title("Flattened ColorHistogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip (chans, ("b", "g", "r")):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        white_pixels += hist[240:256].sum()
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    white_pixels /= 3.0
    total_pixels = corner.shape[0] * corner.shape[1] # row x columns
    colored_pixels = total_pixels - white_pixels
    color_ratio = colored_pixels / total_pixels

    price = 1.25 + color_ratio * (5.0 - 1.25)
    price = round(price, 2)

    print(f"{title} = Price: {price:.2f}")
    return price

#-----------------------------------------#

total_price = 0

for name, corner in quadrants.items():
    price = price_calculator(corner, name)
    total_price += price

print(f"\nTOTAL PRICE for the page = {total_price:.2f} pesos")

#-----------------------------------------#

image[0:450, 0:300] = (225, 0, 0)
image[0:450, 300:600] = (0, 255, 0)
image[450:900, 0:300] = (0, 0, 225)
image[450:900, 300:600] = (225, 0, 225)

cv2.imshow("Quarters", image)
plt.show()
cv2.waitKey(0)
