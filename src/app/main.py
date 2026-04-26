import streamlit as st
import PIL.Image
import requests
import numpy as np
import os
import time
import json

st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")

# MLServer REST Endpoint (Internal to Docker or localhost)
MLSERVER_URL = os.getenv("MLSERVER_URL", "http://localhost:8081/v2/models/pcb-defect-model/infer")

# 1. Inference via MLServer REST API
def predict_via_mlserver(image):
    """
    Sends a request to the MLServer REST API following the V2 Inference Protocol.
    """
    # Preprocess image for MLServer
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
    img_array = np.expand_dims(img_array, axis=0)   # (N, C, H, W)

    # Build V2 Protocol Request
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": img_array.shape,
                "datatype": "FP32",
                "data": img_array.tolist()
            }
        ]
    }

    try:
        response = requests.post(MLSERVER_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse output
        output_data = data['outputs'][0]['data']
        # Simple argmax for binary classification
        label_idx = 1 if output_data[1] > output_data[0] else 0
        confidence = max(output_data) # This is a simplification
        
        return label_idx, confidence, True
    except Exception as e:
        st.error(f"Error connecting to MLServer: {e}")
        return 0, 0.0, False

# --- UI LAYOUT ---
st.title("🔍 PCB Defect Detection Flywheel")
st.write("Professional MLOps Pipeline with MLServer Inference")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Image Capture")
    uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded PCB", use_container_width=True)

with col2:
    st.header("🤖 MLServer Backend")
    if uploaded_file is not None:
        with st.spinner("Requesting Inference from MLServer..."):
            label_idx, confidence, success = predict_via_mlserver(image)
            
            if success:
                defect_found = (label_idx == 1)
                st.metric("Model Confidence", f"{confidence:.2%}")
                
                if defect_found:
                    st.error("❌ Defect Detected!")
                else:
                    st.success("✅ No Defect Detected")
                
                # --- FLYWHEEL LOGIC ---
                if confidence < 0.8:
                    st.warning("⚠️ Low Confidence! Sending to Active Learning Loop.")
                    if st.button("Flag for Human Review"):
                        st.info("Task created in Label Studio.")
                        st.balloons()
            else:
                st.error("❌ MLServer is offline or model not loaded. Start it with `docker-compose up`.")

st.divider()
st.sidebar.title("MLOps Control Panel")
st.sidebar.write("Inference Server: `localhost:8081`")
if st.sidebar.button("Run Retraining (act push)"):
    st.sidebar.write("Simulating local GitHub Action with `act`...")
    time.sleep(3)
    st.sidebar.success("New Model Version Deployed to MLServer!")
