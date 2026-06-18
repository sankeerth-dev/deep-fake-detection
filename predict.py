import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Paths
video_path = r"Dataset/Face swapping/Fake Face swapping/99.mp4"   # ✅ Replace with your input video
output_folder = "ExtractedFaces"
model_path = "resnet18_model.pth"  # ✅ Use your trained model file
os.makedirs(output_folder, exist_ok=True)

# ============================================================
# 1. LOAD YOLO FACE DETECTOR
# ============================================================
face_model = YOLO("yolov11n-face.pt")

# ============================================================
# 2. DEFINE TRANSFORM AND CLASSIFIER
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

class_names = ["fake", "real"]  # same order as training dataset

def get_resnet():
    """Load trained ResNet18 model for binary classification."""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ============================================================
# 3. FACE EXTRACTION
# ============================================================
def extract_faces_from_video(video_path, output_folder, max_faces=20, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Cannot open video: {video_path}")
        return []

    frame_count, saved_faces = 0, 0
    face_paths = []
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened() and saved_faces < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        results = face_model.predict(frame, conf=0.6, verbose=False)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                save_name = f"frame_{frame_count:05d}_{i}.jpg"
                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, face_crop)
                face_paths.append(save_path)
                saved_faces += 1

                if saved_faces >= max_faces:
                    break
            if saved_faces >= max_faces:
                break

        frame_count += 1

    cap.release()
    print(f" Extracted {len(face_paths)} face frames.")
    return face_paths

# ============================================================
# 4. PREDICTION ON EXTRACTED FACES
# ============================================================
def predict_faces(face_paths, model):
    from PIL import Image
    predictions = []
    confidences = []

    for img_path in face_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert BGR → RGB and NumPy → PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # Apply transform and move to device
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.append(pred.item())
            confidences.append(conf.item())

    if len(predictions) == 0:
        print(" No faces detected for prediction.")
        return None, None

    # Majority vote
    most_common = Counter(predictions).most_common(1)[0][0]
    mean_conf = np.mean(confidences)
    final_class = class_names[most_common]

    print(f"\n Prediction Summary:")
    print(f"  → Faces analyzed: {len(predictions)}")
    print(f"  → Majority Vote Class: {final_class.upper()}")
    print(f"  → Mean Confidence: {mean_conf:.3f}")

    return final_class, mean_conf

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n Processing video:", video_path)

    # Step 1: Extract faces
    face_paths = extract_faces_from_video(video_path, output_folder)

    # Step 2: Load classifier and predict
    model = get_resnet()
    result_class, confidence = predict_faces(face_paths, model)

    if result_class:
        print(f"\n Final verdict for {os.path.basename(video_path)}: {result_class.upper()} "
              f"(Confidence: {confidence:.2f})")
