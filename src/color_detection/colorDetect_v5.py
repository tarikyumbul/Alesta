import cv2
import numpy as np

# Load input image
input_image = cv2.imread('src/color_detection/input_images/1.jpg')

# Create a copy of the original image for result
result = input_image.copy()

# Convert image to HSV color space
image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

# Define lower and upper boundaries for red color range
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])
lower2 = np.array([160, 100, 100])
upper2 = np.array([180, 255, 255])

# Create masks for the two ranges of red
lower_mask = cv2.inRange(image_hsv, lower1, upper1)
upper_mask = cv2.inRange(image_hsv, lower2, upper2)

# Combine the masks
full_mask = lower_mask + upper_mask

# Apply the mask to the result image
masked_result = cv2.bitwise_and(result, result, mask=full_mask)

# Convert the mask to grayscale and find contours
gray_mask = cv2.cvtColor(masked_result, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and draw contours based on circularity
for contour in contours:
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # Check if the perimeter is greater than 0 to avoid division by zero
    if perimeter > 0:
        # Approximate the contour to a circle and calculate its circularity
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
        
        # Only draw contours that are sufficiently circular and have a reasonable size
        if 0.7 < circularity < 1.3 and radius > 10:  # You may need to adjust these thresholds
            cv2.circle(result, (int(x), int(y)), int(radius), (0, 255, 0), 2)

# Stack the images horizontally for comparison
stacked_images = np.hstack((input_image, cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR), result))

# Display the stacked images
cv2.imshow('Red Pontoon Detection', cv2.resize(stacked_images, None, fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.destroyAllWindows()
