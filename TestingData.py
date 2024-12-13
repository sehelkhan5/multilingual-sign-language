import cv2
import numpy as np
import os

# Constants for directories, thresholds, and categories
MODE = 'trainingData'
BASE_DIR = f'dataSet/{MODE}/'
MIN_VALUE = 70
CATEGORIES = '0abcdefghijklmnopqrstuvwxyz'

# Initialize video capture
capture = cv2.VideoCapture(0)

# Ensure all category directories exist
for category in CATEGORIES:
    category_dir = os.path.join(BASE_DIR, category.upper())
    os.makedirs(category_dir, exist_ok=True)

# Function to count the number of images in each category
def get_image_count(directory):
    return {char: len(os.listdir(os.path.join(directory, char.upper()))) for char in CATEGORIES}

# Main loop
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Get image counts
    count = get_image_count(BASE_DIR)

    # Display counts dynamically on the frame
    y_offset = 60
    for key, value in count.items():
        cv2.putText(frame, f"{key.upper()} : {value}", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        y_offset += 10

    # Define ROI coordinates
    x1, y1 = int(0.5 * frame.shape[1]), 10
    x2, y2 = frame.shape[1] - 10, int(0.5 * frame.shape[1])

    # Draw ROI on the frame
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Image processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, test_image = cv2.threshold(th3, MIN_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize and display the processed image
    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("Processed Image", test_image)

    # Display the original frame
    cv2.imshow("Frame", frame)

    # Key event handling for data collection
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC key to exit
        break

    if chr(key).lower() in CATEGORIES:
        category = chr(key).upper()
        category_dir = os.path.join(BASE_DIR, category)
        image_path = os.path.join(category_dir, f"{count[chr(key).lower()]}.jpg")
        cv2.imwrite(image_path, roi)

# Release resources
capture.release()
cv2.destroyAllWindows()
