import os
import hashlib
import cv2
import torch
import sqlite3
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from collections import Counter
from PIL import Image
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ============================================================
# DATABASE SETUP (SQLite3)
# ============================================================
st.set_page_config(page_title="DeepFake & Text Detector", layout="wide")
# ================================
# Custom Styling
# ================================
st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#ffbd59"};
            color: {"#FFFFFF"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)
DB_NAME = "users.db"
st.image("coverpage.png")



def hash_password(password: str) -> str:
    """Return SHA-256 hex digest of a plaintext password."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (name TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    conn.commit()
    conn.close()

def register_user(name, password, email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (name, hash_password(password), email))
        conn.commit()
        st.success(" Registration successful! You can now log in.")
    except sqlite3.IntegrityError:
        st.error(" Username already exists. Please choose another.")
    conn.close()

def login_user(name, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name=? AND password=?", (name, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

# ============================================================
# MODEL CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize database
init_db()

# ============================================================
# SESSION STATE MANAGEMENT
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if not st.session_state.logged_in:
    st.markdown("""

## DeepFake and Text Authenticity Detection Platform

The **DeepFake and Text Authenticity Detection System** is an integrated AI-powered solution designed to validate the originality of multimedia content.  
It combines advanced **computer vision** and **natural language processing (NLP)** models to detect tampered, generated, or misleading data across both **video** and **textual** domains.

### 1. System Overview
This platform operates as a dual-modality verification framework:
- **Visual Analysis Module:** Uses **YOLOv11** for rapid face localization and **ResNet18** for authenticity classification of facial frames.
- **Textual Analysis Module:** Employs **GPT-2 perplexity scoring** to assess whether text is likely human-authored or machine-generated.

Each module works independently yet contributes to a unified goal — identifying and mitigating the risks of synthetic media in digital communication, journalism, education, and research.

### 2. Functional Workflow
1. **User Authentication:**  
   Secure access through SQLite3-based registration and login ensures data integrity and session-specific operation.
2. **DeepFake Detection:**  
   Uploaded video files are processed frame by frame. Faces are automatically detected, extracted, and classified as *real* or *fake* with a confidence score.
3. **Text Authenticity Check:**  
   Text input is evaluated using GPT-2’s language modeling capability to compute a **perplexity score**, a statistical measure of how predictable the text is. Lower scores imply AI-generated text; higher scores indicate natural human writing.
4. **Result Interpretation:**  
   Outputs are clearly visualized with dynamic color-coded verdicts (green for authentic content, red for synthetic).

### 3. System Features
- **Secure Authentication:** User credentials stored locally via SQLite3 with session state tracking.
- **GPU-Optimized Inference:** Efficient PyTorch-based deployment supporting CUDA acceleration.
- **Streamlit Interface:** Clean, interactive, and responsive dashboard for real-time analysis.
- **Data Privacy:** No cloud storage; all operations performed locally within session scope.
- **Explainable Results:** Confidence metrics and model decisions displayed transparently.

### 4. Real-World Relevance
The exponential rise of AI-generated videos and text threatens digital credibility and public trust.  
This system addresses that by combining two core forensic dimensions — *visual evidence integrity* and *linguistic authenticity* — providing a scalable solution for institutions, journalists, educators, and content moderators to safeguard the truthfulness of multimedia data.

---

### About the Developer
**Developed by:** 
**Division:** AI & Machine Learning Research  
**Focus Areas:** Deep Learning, Vision-Language Modeling, Synthetic Media Detection  
""")
# ============================================================
# AUTHENTICATION SECTION
# ============================================================
st.sidebar.title(" Secure Login Portal")
st.sidebar.image("s1.png")
st.sidebar.markdown("""
### Project Overview
**DeepFake and Text Authenticity Detection System**  
A dual-modality AI platform for verifying the originality of multimedia content.

---

#### Vision Module
- **YOLOv11:** Advanced face localization and segmentation.  
- **ResNet18:** Binary classifier distinguishing real vs. fake facial patterns.  
- **Confidence Visualization:** Displays per-frame and aggregate classification confidence.

#### Language Module
- **GPT-2 Perplexity Analysis:** Measures text predictability to identify AI-generated or synthetic writing.  
- **Adaptive Thresholding:** Interprets perplexity scores into human vs. AI probability.

---

#### Core Features
- Local authentication via **SQLite3**
- **Session-based login** for user state management
- **Offline-capable**—no external API dependencies
- GPU/CPU **auto-detection** for fast inference
- Modular structure for easy future upgrades

---

#### Technical Stack
- Python, Streamlit  
- PyTorch and TorchVision  
- Ultralytics YOLOv11  
- Hugging Face Transformers  
- SQLite3 Database  

---

#### Mission
To enhance **digital trust** by combining vision and language intelligence for detecting and preventing **AI-generated misinformation**.
""")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if not st.session_state.logged_in:
    if choice == "Login":
        st.sidebar.subheader("Login to Continue")
        name = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login_user(name, password):
                st.session_state.logged_in = True
                st.session_state.username = name
                st.sidebar.success(f" Welcome, {name}! You are now logged in.")
            else:
                st.sidebar.error(" Invalid username or password.")
    elif choice == "Register":
        st.subheader("Register New Account")
        name = st.text_input("Choose a Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if name and password and email:
                register_user(name, password, email)
            else:
                st.warning("Please fill out all fields to register.")

# ============================================================
# POST-LOGIN DASHBOARD
# ============================================================
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.markdown("---")
    app_choice = st.sidebar.selectbox(
        "Select Feature:",
        ["-- Select --", "DeepFake Video Detection", "Text Authenticity Check"]
    )

    # Paths and setup
    output_folder = "ExtractedFaces"
    model_path = "resnet18_model.pth"  # Your trained ResNet18 model
    os.makedirs(output_folder, exist_ok=True)

    # --------------------- Load YOLO Face Detector ---------------------
    face_model = YOLO("yolov11n-face.pt")

    # --------------------- Define Transform ---------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    class_names = ["fake", "real"]

    def get_resnet():
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # --------------------- Face Extraction ---------------------
    def extract_faces_from_video(video_path, output_folder, max_faces=20, frame_skip=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Cannot open video: {video_path}")
            return []

        frame_count, saved_faces = 0, 0
        face_paths = []
        os.makedirs(output_folder, exist_ok=True)

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
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
            progress.progress(min(frame_count / total_frames, 1.0))
        cap.release()
        st.success(f" Extracted {len(face_paths)} face frames.")
        return face_paths

    # --------------------- Prediction ---------------------
    def predict_faces(face_paths, model):
        predictions, confidences = [], []

        for img_path in face_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                predictions.append(pred.item())
                confidences.append(conf.item())

        if not predictions:
            st.warning("No faces detected for prediction.")
            return None, None

        most_common = Counter(predictions).most_common(1)[0][0]
        mean_conf = np.mean(confidences)
        final_class = class_names[most_common]

        st.subheader("Prediction Summary")
        st.write(f"**Faces analyzed:** {len(predictions)}")

        # ======= Red/Green Result Display =======
        if final_class.lower() == "fake":
            st.markdown(
                f"<h2 style='color:red; font-size:38px;'> FINAL VERDICT: FAKE VIDEO (Confidence: {mean_conf:.2f})</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<h2 style='color:green; font-size:38px;'> FINAL VERDICT: REAL VIDEO (Confidence: {mean_conf:.2f})</h2>",
                unsafe_allow_html=True,
            )

        return final_class, mean_conf

    # --------------------- GPT-2 Text Auth ---------------------
    @st.cache_resource
    def load_gpt2_model():
        model_name = "gpt2"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model.eval()
        return model, tokenizer

    def compute_perplexity(text, model, tokenizer):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    # ============================================================
    # MODE HANDLING
    # ============================================================
    if app_choice == "DeepFake Video Detection":
        st.header(" DeepFake Video Detection")
        uploaded_file = st.file_uploader("Upload a video (MP4/MOV/AVI):", type=["mp4", "mov", "avi"])
        if uploaded_file:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(video_path)
            if st.button("Detect DeepFake"):
                st.info("Processing video...")
                face_paths = extract_faces_from_video(video_path, output_folder)
                model = get_resnet()
                result_class, confidence = predict_faces(face_paths, model)
                if result_class:
                    if result_class == "fake":
                        st.error(f"️ DeepFake Detected (Confidence: {confidence:.2f})")
                    else:
                        st.success(f" Real Video (Confidence: {confidence:.2f})")

    elif app_choice == "Text Authenticity Check":
        st.header(" Text Authenticity Check (GPT-2 Perplexity)")
        text_input = st.text_area("Paste text here:", height=200)
        if st.button("Analyze Text"):
            if not text_input.strip():
                st.warning("Please enter text to analyze.")
            else:
                st.info("Computing perplexity using GPT-2...")
                model, tokenizer = load_gpt2_model()
                score = compute_perplexity(text_input, model, tokenizer)
                st.write(f"**Perplexity Score:** {score:.2f}")
                if score < 40:
                    st.error(" Likely AI-generated (Low perplexity).")
                elif 40 <= score < 80:
                    st.warning(" Possibly mixed AI + human content.")
                else:
                    st.success(" Likely human-written (High perplexity).")
