# Deepfake Detection + Explanation System

An AI-powered forensic platform that detects manipulated images/videos, highlights suspicious regions, and generates human-readable explanations using Explainable AI and LLMs.

---

## 🚀 Features

- Deepfake detection for images and videos
- CNN-based classification using Xception / EfficientNet / ResNet
- Face detection and preprocessing pipeline
- Grad-CAM explainability heatmaps
- LLM-generated forensic explanations
- Video frame-level analysis
- Interactive dashboard for visualization
- PDF forensic report generation

---

## 🎯 Project Goal

The system can:

✅ Detect whether media is fake or real  
✅ Highlight manipulated facial regions  
✅ Explain why the content is suspicious  
✅ Generate AI-based forensic reasoning  

### Example Output

> “The video is likely manipulated because facial blending artifacts and inconsistent eye reflections were detected around the mouth and cheek regions.”

---

## 🧠 System Architecture

```text
User Upload (Image / Video)
        ↓
Frame Extraction
        ↓
Face Detection
        ↓
CNN Deepfake Detector
        ↓
Prediction (Fake / Real)
        ↓
Grad-CAM Explainability
        ↓
LLM Explanation Generator
        ↓
Final Visualization & Report
```

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI

### Deep Learning
- PyTorch / TensorFlow
- Xception
- EfficientNet
- ResNet

### Computer Vision
- OpenCV
- MTCNN / RetinaFace

### Explainability
- Grad-CAM

### Frontend
- Streamlit / React

### LLM Integration
- OpenAI API / Local LLMs

---

## 📂 Datasets

- FaceForensics++
- Celeb-DF
- DFDC (DeepFake Detection Challenge)


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/deepfake-intelligence.git

cd deepfake-intelligence

pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Backend
```bash
uvicorn api.main:app --reload
```

### Frontend
```bash
streamlit run frontend/app.py
```

---

## 📊 Features Included

- Image deepfake detection
- Video deepfake detection
- Frame aggregation analysis
- Suspicious region highlighting
- AI-generated explanations
- Visualization dashboard
- Explainable AI integration

---

## 🔥 Advanced Extensions

- Audio deepfake detection
- Multi-modal analysis
- Temporal consistency checking
- SHAP / Attention-map explainability
- Adversarial robustness testing

---

## 💼 Resume Bullet

> Built an explainable deepfake detection system using CNN-based computer vision models, Grad-CAM visualizations, and LLM-generated forensic explanations for manipulated media analysis.

---


## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
