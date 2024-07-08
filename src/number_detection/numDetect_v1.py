import easyocr
import cv2
import numpy as np

# Function to draw bounding box and label text on image
def draw_boxes(image, bounds, color=(0, 255, 255), width=2):
    for bound in bounds:
        p1, p2, p3, p4 = bound[0]
        p1 = int(p1[0]), int(p1[1])
        p2 = int(p2[0]), int(p2[1])
        p3 = int(p3[0]), int(p3[1])
        p4 = int(p4[0]), int(p4[1])
        cv2.line(image, p1, p2, color, width)
        cv2.line(image, p2, p3, color, width)
        cv2.line(image, p3, p4, color, width)
        cv2.line(image, p4, p1, color, width)
        cv2.putText(image, bound[1], (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Load image
input_image = cv2.imread('src/number_detection/input_images/Screenshot_1.png')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Perform OCR on the image
result = reader.readtext(input_image)

# Draw bounding boxes and labels on the image
output_image = input_image.copy()
draw_boxes(output_image, result)

# Display the output image
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
