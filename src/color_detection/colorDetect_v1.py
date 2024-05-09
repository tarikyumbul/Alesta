import cv2
import numpy as np

def detect_color(frame, lower_bound, upper_bound):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range to detect
    lower_bound = np.array(lower_bound, dtype=np.uint8)
    upper_bound = np.array(upper_bound, dtype=np.uint8)
    
    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    # Bitwise AND to extract the color from the original frame
    color_extracted = cv2.bitwise_and(frame, frame, mask=mask)
    
    return color_extracted

# Define the lower and upper bounds of the color (here we are detecting red)
lower_red = [0, 100, 100]
upper_red = [10, 255, 255]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame from webcam")
        break
    
    # Detect the color
    color_detected = detect_color(frame, lower_red, upper_red)
    
    # Display the color-detected frame
    cv2.imshow('Color Detected', color_detected)
    
    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
