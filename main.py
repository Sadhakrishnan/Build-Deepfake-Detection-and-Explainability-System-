import sys
import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import torch
import base64

from preprocessing.face_detector import FaceDetector
from preprocessing.frame_extractor import extract_frames
from models.detector import DeepfakeDetector
from gradcam import GradCAM, overlay_heatmap
from explainer import generate_explanation

app = FastAPI(title="Deepfake Detection API")

# Initialize models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_detector = FaceDetector(device=device)

# Load deepfake detector
detector_model = DeepfakeDetector(pretrained=True).to(device)
detector_model.eval()

# Initialize Grad-CAM
# Target the last convolutional layer of ResNet50
target_layer = detector_model.model.layer4[-1].conv3
grad_cam = GradCAM(detector_model.model, target_layer)

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Detect and crop face
    face_tensor = face_detector.detect_and_crop(image_rgb)
    
    if face_tensor is None:
        return JSONResponse({"error": "No face detected in the image."}, status_code=400)
        
    face_tensor = face_tensor.to(device)
    
    # 2. Predict Deepfake probability
    fake_probability = detector_model.predict(face_tensor)
    prediction = "Fake" if fake_probability > 0.5 else "Real"
    
    # 3. Generate Grad-CAM heatmap
    if len(face_tensor.shape) == 3:
        input_tensor = face_tensor.unsqueeze(0)
    else:
        input_tensor = face_tensor
        
    heatmap = grad_cam(input_tensor, class_idx=0)
    
    # Reverse normalization if applied
    face_img = face_tensor.cpu().numpy()
    if face_img.shape[0] == 3:
        face_img = np.transpose(face_img, (1, 2, 0))
    face_img = (face_img * 255).astype(np.uint8)
    
    overlay_img = overlay_heatmap(face_img, heatmap)
    
    # Encode overlay image to base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 4. Generate explanation
    suspicious_regions = ["mouth blending", "eye inconsistencies"] if fake_probability > 0.5 else []
    explanation = generate_explanation(fake_probability, suspicious_regions)
    
    return {
        "prediction": prediction,
        "fake_probability": float(fake_probability),
        "explanation": explanation,
        "heatmap_image_base64": overlay_base64
    }

@app.post("/video_detect")
async def detect_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())
        
    try:
        frames, metadata = extract_frames(temp_video_path, frame_interval=15)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    if not frames:
        return JSONResponse({"error": "Failed to extract frames or video is empty."}, status_code=400)
        
    frame_results = []
    fake_count = 0
    
    for idx, frame in enumerate(frames):
        face_tensor = face_detector.detect_and_crop(frame)
        if face_tensor is not None:
            face_tensor = face_tensor.to(device)
            fake_prob = detector_model.predict(face_tensor)
            
            is_fake = fake_prob > 0.5
            if is_fake:
                fake_count += 1
                
            frame_results.append({
                "frame_id": metadata[idx]["frame_id"],
                "timestamp": metadata[idx]["timestamp"],
                "fake_probability": float(fake_prob),
                "prediction": "Fake" if is_fake else "Real"
            })
            
    if not frame_results:
         return JSONResponse({"error": "No faces detected in the video."}, status_code=400)
         
    video_fake_prob = fake_count / len(frame_results)
    final_decision = "Fake" if video_fake_prob > 0.5 else "Real"
    
    return {
        "aggregated_decision": final_decision,
        "video_fake_probability": float(video_fake_prob),
        "frame_results": frame_results
    }
