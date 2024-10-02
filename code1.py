import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import Tracker
from datetime import datetime

model = YOLO("yolov10s.pt")

# Global variables to store line points
line_points = []
line_drawn = False
crossed_ids = set()

# Function to calculate which side of the line a point is on
def get_line_side(point, line_point1, line_point2):
    return (point[0] - line_point1[0]) * (line_point2[1] - line_point1[1]) - (point[1] - line_point1[1]) * (line_point2[0] - line_point1[0])

def draw_line(event, x, y, flags, param):
    global line_points, line_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
        if len(line_points) == 2:
            line_drawn = True  # Mark the line as drawn

# Set up mouse callback to draw the line
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', draw_line)

cap = cv2.VideoCapture('footage.mov')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker = Tracker()
count = 0

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
list1 = []

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    processed_frame, detected_label = process_frame(frame)
    print(detected_label)

    # Draw the manually placed line
    if line_drawn and len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 255, 255), 3)

    # Object detection and tracking
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    detected_boxes = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]
        detected_boxes.append([x1, y1, x2, y2])  # Ensure this has exactly four values

    # Update tracker with the bounding boxes
    bbox_idx = tracker.update(detected_boxes)

    if line_drawn and len(line_points) == 2:
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            cx = int((x3 + x4) // 2)  # Calculate center of the bounding box
            cy = int((y3 + y4) // 2)

            # Check if this ID has been seen before
            if id in crossed_ids:
                continue

            # Get which side of the line the current center of the car is on
            current_position = (cx, cy)
            current_side = get_line_side(current_position, line_points[0], line_points[1])

            # If the car crosses from one side to the other, it has crossed the line
            if current_side < 0:  # Check if the car crosses the line
                if 'car' in c and detected_label == "RED":
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                    # Save the image with red label
                    timestamp = datetime.now().strftime('%H-%M-%S')
                    image_filename = f"{timestamp}.jpg"
                    output_path = os.path.join(output_dir, image_filename)

                    if list1.count(id) == 0:
                        list1.append(id)
                        cv2.imwrite(output_path, frame)

                    # Add the ID to the set of crossed cars
                    crossed_ids.add(id)
                else:
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    cv2.imshow("RGB", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
