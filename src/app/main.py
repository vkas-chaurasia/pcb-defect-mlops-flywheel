import streamlit as st
import PIL.Image
import random
import time

st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")

st.title("🔍 PCB Defect Detection Flywheel")
st.write("Upload a PCB image to detect defects and trigger the Active Learning loop.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Image Capture")
    uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Uploaded PCB", use_container_width=True)

with col2:
    st.header("🤖 Model Inference")
    if uploaded_file is not None:
        with st.spinner("Analyzing..."):
            time.sleep(1.5)  # Simulate processing
            
            # Placeholder for actual model inference
            confidence = random.uniform(0.4, 0.95)
            defect_found = random.choice([True, False])
            
            st.metric("Confidence Score", f"{confidence:.2%}")
            
            if defect_found:
                st.error("❌ Defect Detected!")
            else:
                st.success("✅ No Defect Detected")
            
            # Flywheel Logic
            if confidence < 0.8:
                st.warning("⚠️ Low Confidence! Sending to Label Studio for human review.")
                if st.button("Manual Review & Add to Flywheel"):
                    st.info("Image synced to Label Studio. Retraining pipeline triggered.")
                    st.balloons()
            else:
                st.info("✅ High Confidence. Automatically saving to Production DB.")

st.divider()
st.sidebar.title("MLOps Control Panel")
st.sidebar.info("System Status: Local Orchestration Active")
if st.sidebar.button("Run Retraining (DVC repro)"):
    st.sidebar.write("Executing `dvc repro`...")
    time.sleep(2)
    st.sidebar.success("Model Updated Successfully!")
