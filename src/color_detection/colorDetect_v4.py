import cv2
import numpy as np

# Load input image
input_image = cv2.imread('src/color_detection/input_images/3.jpg')

# Create a copy of the original image for result
result = input_image.copy()

# Convert image to HSV color space
image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

def take_input():
    color = input("Enter the color to be detected: ")

    if color == "red":
        # Define lower and upper boundaries for red color range
        take_input.lower1 = np.array([0, 100, 100])    # Red: [0, 100, 20]    //  Green: [36, 50, 50]    //  Yellow: [20, 100, 100]
        take_input.upper1 = np.array([20, 255, 255])   # Red: [10, 255, 255]  //  Green: [86, 255, 255]  //  Yellow: [30, 255, 255]

        take_input.lower2 = np.array([160,100,100])    # Red: [160,100,20]    //  Green: [36, 50, 50]    //  Yellow: [20, 100, 100]
        take_input.upper2 = np.array([180,255,255])    # Red: [179,255,255]   //  Green: [86, 255, 255]  //  Yellow: [30, 255, 255]

    elif color == "green":
        take_input.lower1 = np.array([36, 50, 50])
        take_input.upper1 = np.array([86, 255, 255])

        take_input.lower2 = np.array([36, 50, 50])
        take_input.upper2 = np.array([86, 255, 255])

    elif color == "yellow":
        take_input.lower1 = np.array([20, 100, 100])
        take_input.upper1 = np.array([30, 255, 255])

        take_input.lower2 = np.array([20, 100, 100])
        take_input.upper2 = np.array([30, 255, 255])

    else:
        print("Invalid input.")
        take_input()

take_input()

# Create masks for the two ranges of red
lower_mask = cv2.inRange(image_hsv, take_input.lower1, take_input.upper1)
upper_mask = cv2.inRange(image_hsv, take_input.lower2, take_input.upper2)

# Combine the masks
full_mask = lower_mask + upper_mask

# Apply the mask to the result image
result = cv2.bitwise_and(result, result, mask=full_mask)

# Stack the images horizontally
stacked_images = np.hstack((input_image, cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR), result))

# Display the stacked images
cv2.imshow('Color Detection', cv2.resize(stacked_images, None, fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.destroyAllWindows()
