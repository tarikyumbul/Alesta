import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Take each frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame from webcam")
        break
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # Define range of red color in HSV
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])  
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
 
    # Convert mask to 3-channel image for stacking
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 
    # Stack up all three images together
    stacked = np.hstack((mask_3, frame, res))
     
    # Display the result
    cv2.imshow('Result', cv2.resize(stacked, None, fx=0.8, fy=0.8))
    
    # Check for 'Esc' key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
