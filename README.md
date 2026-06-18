# 🎭 DeepFake & Text Authenticity Detection System

> An AI-powered dual-modality platform for detecting deepfake videos and AI-generated text — built with PyTorch, YOLOv11, ResNet18, and GPT-2.

---

## 📸 Preview

![Cover](coverpage.png)

---

## 🚀 Features

- 🎥 **DeepFake Video Detection** — Frame-by-frame face extraction using YOLOv11 + ResNet18 binary classifier
- 🧠 **Text Authenticity Check** — GPT-2 perplexity scoring to detect AI-generated content
- 🔐 **Secure Login System** — SQLite3-based user authentication with hashed passwords
- ⚡ **GPU/CPU Auto-Detection** — CUDA-accelerated inference via PyTorch
- 🖥️ **Interactive Dashboard** — Built with Streamlit for real-time analysis
- 🔒 **Fully Offline** — No cloud API dependencies; all processing happens locally

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Deep Learning | PyTorch, TorchVision |
| Face Detection | Ultralytics YOLOv11 |
| Classification | ResNet18 (fine-tuned) |
| Text Analysis | Hugging Face GPT-2 |
| Database | SQLite3 |
| Containerization | Docker |

---

## 📁 Project Structure

```
DeepFake-Detection/
├── app.py                  # Main Streamlit application
├── predict.py              # Standalone prediction script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container config
├── Deep Fake detection.ipynb  # Training & experimentation notebook
├── coverpage.png           # UI cover image
├── s1.png                  # Sidebar image
└── .gitignore
```

> ⚠️ **Model files (`.pth`) and datasets are NOT included** in this repo due to size constraints. See [Model Setup](#-model-setup) below.

---

## ⚙️ Installation

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model files (see Model Setup section)

# 5. Run the app
streamlit run app.py
```

### Option 2: Docker

```bash
docker build -t deepfake-detector .
docker run -p 8501:8501 deepfake-detector
```
Then open [http://localhost:8501](http://localhost:8501)

---

## 🤖 Model Setup

The trained model files are hosted separately due to GitHub's file size limits.

| Model | Description | Download |
|---|---|---|
| `resnet18_model.pth` | ResNet18 deepfake classifier | *(add your link)* |
| `cnn_model.pth` | CNN deepfake classifier | *(add your link)* |
| `vgg16_model.pth` | VGG16 deepfake classifier | *(add your link)* |
| `yolov11n-face.pt` | YOLOv11 face detector | *(add your link)* |

After downloading, place all `.pth` / `.pt` files in the **root project directory**.

---

## 🔍 How It Works

### DeepFake Video Detection
1. User uploads an MP4/MOV/AVI video
2. YOLOv11 detects and crops faces frame-by-frame
3. Each face is passed through ResNet18 for real/fake classification
4. Confidence scores are aggregated → final verdict displayed

### Text Authenticity Detection
1. User pastes text into the input box
2. GPT-2 computes the **perplexity score** of the text
3. Low perplexity → likely AI-generated | High perplexity → likely human-written

| Perplexity Score | Verdict |
|---|---|
| < 40 | 🤖 Likely AI-generated |
| 40 – 80 | ⚠️ Mixed / Uncertain |
| > 80 | ✅ Likely human-written |

---

## 🔐 Security

- Passwords are hashed using **SHA-256** before storage
- Session state managed per-user via Streamlit's `session_state`
- No data leaves the local machine

---

## 📊 Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| ResNet18 | ~XX% | Trained on FaceForensics++ |
| CNN | ~XX% | Custom architecture |
| VGG16 | ~XX% | Transfer learning |

> Fill in your actual accuracy values from your notebook results.

---

## 🐳 Docker Support

This project includes a Dockerfile for containerized deployment. See [Installation → Docker](#option-2-docker).

---

## 🛡️ License

This project is for academic and research purposes only.

---

## 👨‍💻 Developer

**Developed by:** *(Your Name)*  
**Division:** AI & Machine Learning Research  
**Focus Areas:** Deep Learning, Vision-Language Modeling, Synthetic Media Detection
