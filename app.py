import streamlit as st
import requests
import io
from PIL import Image
import base64

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Deepfake Forensic Dashboard", layout="wide")

st.title("🕵️‍♂️ Deepfake Detection & Explainability System")
st.write("Upload an image or video to analyze it for deepfake manipulation.")

st.sidebar.title("Settings")
analysis_type = st.sidebar.radio("Analysis Type", ["Image Analysis", "Video Analysis"])

if analysis_type == "Image Analysis":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing media..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/detect", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.subheader("Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", result["prediction"])
                        with col2:
                            st.metric("Fake Probability", f"{result['fake_probability']*100:.2f}%")
                            
                        st.subheader("Suspicious Regions (Grad-CAM)")
                        heatmap_bytes = base64.b64decode(result["heatmap_image_base64"])
                        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                        st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                        
                        st.subheader("AI Forensic Explanation")
                        st.info(result["explanation"])
                    else:
                        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

elif analysis_type == "Video Analysis":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Analyze Video"):
            with st.spinner("Extracting frames and analyzing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/video_detect", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.subheader("Video Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Aggregated Decision", result["aggregated_decision"])
                        with col2:
                            st.metric("Overall Fake Probability", f"{result['video_fake_probability']*100:.2f}%")
                            
                        st.subheader("Timeline Analysis")
                        
                        # Simple timeline visualization
                        for frame in result["frame_results"]:
                            st.write(f"**{frame['timestamp']}** (Frame {frame['frame_id']}): {frame['prediction']} - {frame['fake_probability']*100:.2f}%")
                            if frame['prediction'] == "Fake":
                                st.progress(frame['fake_probability'])
                    else:
                         st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
