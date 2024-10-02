import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import *
from datetime import datetime

# Load YOLO model for vehicle detection
model = YOLO("yolov10s.pt")

# Load Haar cascade for license plate detection
license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")


# Define mouse callback to track points (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Capture the video
cap = cv2.VideoCapture('tech.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker = Tracker()
count = 0
area = [(324, 313), (283, 374), (854, 392), (864, 322)]  # Define the area (stop line region)

# Create directory for saving images with today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create directory for saving license plates on desktop
license_plate_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Captured_License_Plates")
if not os.path.exists(license_plate_dir):
    os.makedirs(license_plate_dir)

list1 = []


# Function to extract license plates
def extract_license_plate(frame, mask_line):
    # Convert the image to grayscale (Haar cascades are typically trained on grayscale images)
    gray = cv2.cvtColor(mask_line, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to equalize the histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Erode the image using a 2x2 kernel to remove noise
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)

    # Find the bounding box of non-black pixels
    non_black_points = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(non_black_points)

    # Calculate the new width of the bounding box, excluding 30% on the right side
    w = int(w * 0.7)

    # Crop the image to the bounding box
    cropped_gray = gray[y:y + h, x:x + w]

    # Detect license plates in the image
    license_plates = license_plate_cascade.detectMultiScale(cropped_gray, scaleFactor=1.07, minNeighbors=15,
                                                            minSize=(20, 20))

    # List to hold cropped license plate images
    license_plate_images = []

    # Loop over the detected license plates
    for (x_plate, y_plate, w_plate, h_plate) in license_plates:
        # Draw a rectangle around the license plate in the original frame
        cv2.rectangle(frame, (x_plate + x, y_plate + y), (x_plate + x + w_plate, y_plate + y + h_plate), (0, 255, 0), 3)

        # Crop the license plate and append to list
        license_plate_image = cropped_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]
        license_plate_images.append(license_plate_image)

    return frame, license_plate_images


# Main loop
while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame, detected_label = process_frame(frame)
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]
        list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox

        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)

        # Check if vehicle is in the stop line area and the light is red
        if result >= 0:
            if 'car' in c and detected_label == "RED":
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                # Save the image of the red light violation
                timestamp = datetime.now().strftime('%H-%M-%S')
                image_filename = f"{timestamp}.jpg"
                output_path = os.path.join(output_dir, image_filename)
                if list1.count(id) == 0:
                    list1.append(id)
                    cv2.imwrite(output_path, frame)

                    # Extract the license plate from the car
                    car_image = frame[y3:y4, x3:x4]
                    _, license_plates = extract_license_plate(frame, car_image)
                    if license_plates:
                        for i, plate in enumerate(license_plates):
                            plate_filename = f"{timestamp}_plate_{i}.jpg"
                            plate_output_path = os.path.join(license_plate_dir, plate_filename)
                            cv2.imwrite(plate_output_path, plate)
            else:
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    # Draw the stop line on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
