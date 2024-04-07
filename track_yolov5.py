import cv2
import numpy as np
from sort import Sort
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect(model, img):
    # Perform detection
    results = model(img)
    # Extract detections
    detections = results.xyxy[0]
    # Convert detections to required format: [[x1, y1, x2, y2, confidence], ...]
    formatted_detections = detections.cpu().numpy()
    return formatted_detections[:, :5]

# Path to the folder containing images
images_folder_path = '/home/kunj/Downloads/cv/MOT15/train/PETS09-S2L1/img1'  # Adjust this path to your images folder

# Initialize SORT tracker
tracker = Sort()

# Prepare to save the tracker's output
tracker_outputs = []

import os
if not os.path.exists('output_frames'):
    os.makedirs('output_frames')

frame_count = 0
# Load images and process each frame
for img_file in sorted(Path(images_folder_path).glob('*.jpg')):
    frame = cv2.imread(str(img_file))

    # Use YOLOv5 to detect objects in the current frame
    dets = detect(model, frame)

    # Update tracker with current frame detections
    trackers = tracker.update(dets)

    # Collect tracker outputs for saving and draw tracking results
    for d in trackers:
        d = d.astype(np.int32)
        tracker_id = int(d[4])
        bbox = (d[0], d[1], d[2] - d[0], d[3] - d[1])  # Convert to x, y, width, height
        tracker_outputs.append((img_file.stem, tracker_id, *bbox))  # Include frame_number and tracker_id

        # Optionally draw tracking results on the frame
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255, 255, 255), 2)
        cv2.putText(frame, str(tracker_id), (d[0], d[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with tracking results
    output_filename = f'output_frames/frame_{frame_count}.jpg'
    cv2.imwrite(output_filename, frame)
    cv2.imshow('Frame', frame)
    frame_count += 1

    # Break the loop when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the video capture and window
cv2.destroyAllWindows()

# Save the tracker's output to a file
with open('pets09', 'w') as f:
    for line in tracker_outputs:
        # Format: frame_number, tracker_id, x, y, w, h, 1, -1, -1, -1
        f.write(','.join(map(str, line)) + ",1,-1,-1,-1\n")
