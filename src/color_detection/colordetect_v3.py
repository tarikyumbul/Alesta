import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame from webcam")
        break
    
    # Create a copy of the original frame for result
    result = frame.copy()

    # Convert frame to HSV color space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper boundaries for red color range
    lower1 = np.array([0, 100, 100])     # Red: [0, 100, 20]    //  Green: [36, 50, 50]    //  Yellow: [20, 100, 100]
    upper1 = np.array([20, 255, 255])   # Red: [10, 255, 255]  //  Green: [86, 255, 255]  //  Yellow: [30, 255, 255]

    lower2 = np.array([160,100,100])     # Red: [160,100,20]    //  Green: [36, 50, 50]    //  Yellow: [20, 100, 100]
    upper2 = np.array([180,255,255])    # Red: [179,255,255]   //  Green: [86, 255, 255]  //  Yellow: [30, 255, 255]

    # Create masks for the two ranges of red
    lower_mask = cv2.inRange(frame_hsv, lower1, upper1)
    upper_mask = cv2.inRange(frame_hsv, lower2, upper2)

    # Combine the masks
    full_mask = lower_mask + upper_mask

    # Apply the mask to the result frame
    result = cv2.bitwise_and(result, result, mask=full_mask)

    # Stack the frames horizontally
    stacked_frames = np.hstack((frame, cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR), result))

    # Display the stacked frames
    cv2.imshow('Color Detection', cv2.resize(stacked_frames, None, fx=0.8, fy=0.8))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
