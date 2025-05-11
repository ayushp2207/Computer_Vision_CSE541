import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sort import Sort
from sklearn.metrics.pairwise import cosine_similarity

# Load YOLOv5
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load a pre-trained ResNet50 model for feature extraction
resnet_model = resnet50(pretrained=True)
resnet_model.eval()  # Set to evaluation mode

# Preprocessing for ResNet50
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_embeddings(frame, detections):
    embeddings = []
    for det in detections:
        # Crop and preprocess the image
        x1, y1, x2, y2 = map(int, det[:4])
        crop_img = frame[y1:y2, x1:x2]
        crop_img = Image.fromarray(crop_img)
        input_tensor = preprocess(crop_img)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Generate embedding
        with torch.no_grad():
            output = resnet_model(input_batch)
        
        # Normalize the output to get the embedding
        norm_output = torch.nn.functional.normalize(output[0], dim=0)
        embeddings.append(norm_output.numpy())
    return np.array(embeddings)

def cosine_similarity_n_space(m1, m2, threshold=0.99):
    # m1 and m2 are two matrices with vectors to compare pairwise.
    # Each vector is normalized to have unit L2 norm before comparison.
    m1 = m1 / np.linalg.norm(m1, axis=1, keepdims=True)
    m2 = m2 / np.linalg.norm(m2, axis=1, keepdims=True)
    sim = np.dot(m1, m2.T)
    max_sim = np.max(sim, axis=1)
    max_sim_ids = np.argmax(sim, axis=1)
    return [(i, max_sim_ids[i]) for i in range(len(max_sim)) if max_sim[i] > threshold]

# Initialize SORT tracker
tracker = Sort()

# Mapping from SORT tracker IDs to ResNet embeddings
tracked_embeddings = {}

images_folder_path = '/home/kunj/Downloads/cv/MOT15/train/PETS09-S2L1/img1'

tracker_outputs = []

for img_file in sorted(Path(images_folder_path).glob('*.jpg')):
    frame = cv2.imread(str(img_file))
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Assuming we're only interested in the first image
    embeddings = extract_embeddings(frame, detections)

    # Update tracker with detections
    trackers = tracker.update(detections[:, :4])

    new_tracked_embeddings = {}
    for tracker_info, embedding in zip(trackers, embeddings):
        x1, y1, x2, y2, obj_id = tracker_info.astype(int)
        if len(tracked_embeddings) > 0:
            obj_embeddings = np.array(list(tracked_embeddings.values()))
            matches = cosine_similarity_n_space(np.array([embedding]), obj_embeddings)

            if matches:
                # Object re-identified, update the tracker ID to the matched ID.
                _, matched_id = matches[0]
                obj_id = list(tracked_embeddings.keys())[matched_id]

        new_tracked_embeddings[obj_id] = embedding  # Update the tracker embeddings

        # Draw bounding boxes and IDs on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Update the tracked embeddings with the latest frame information
    tracked_embeddings = new_tracked_embeddings

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
