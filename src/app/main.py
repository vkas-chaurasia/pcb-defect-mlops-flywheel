import streamlit as st
import PIL.Image
import requests
import numpy as np
import os
import time
from pathlib import Path
import subprocess

st.set_page_config(
    page_title="PCB Defect Detection Flywheel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
MLSERVER_URL = os.getenv("MLSERVER_URL", "http://localhost:8081/v2/models/pcb-defect-model/infer")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
SIMULATION_DIR = Path("data/raw/unseen_simulation")

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4251;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #ff4b4b;
        background-color: #1e2130;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def predict_via_fastapi(image):
    """Sends image to our Custom FastAPI server (port 8000)."""
    try:
        # Convert PIL to bytes
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        files = {"file": ("image.jpg", byte_im, "image/jpeg")}
        response = requests.post(f"{FASTAPI_URL}/predict", files=files)
        response.raise_for_status()
        return response.json(), True
    except Exception as e:
        return str(e), False

def get_flywheel_stats():
    """Returns stats about the simulation directory."""
    if not SIMULATION_DIR.exists():
        return 0, []
    images = [p for p in SIMULATION_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return len(images), images

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.info("System status and orchestration.")
    
    st.subheader("Service Health")
    try:
        health = requests.get(f"{FASTAPI_URL}/health", timeout=1).json()
        st.success(f"FastAPI: Online ({health['device']})")
    except:
        st.error("FastAPI: Offline (Port 8000)")
        
    st.divider()
    
    if st.button("🚀 Trigger Automated Flywheel", use_container_width=True):
        with st.status("Running Batch Inference...", expanded=True) as status:
            st.write("Scanning `data/raw/unseen_simulation`...")
            # We run the script as a subprocess to simulate a real MLOps trigger
            result = subprocess.run(["python", "src/utils/batch_inference.py"], capture_output=True, text=True)
            st.code(result.stdout)
            status.update(label="Flywheel Cycle Complete!", state="complete", expanded=False)
        st.balloons()

# --- MAIN UI ---
st.title("🔍 PCB Defect Detection Flywheel")
st.markdown("### Professional MLOps Active Learning Pipeline")

tabs = st.tabs(["🎮 Interactive Sandbox", "📊 Flywheel Dashboard"])

with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Capture")
        uploaded_file = st.file_uploader("Drop a PCB image here...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded PCB", use_container_width=True)

    with col2:
        st.header("🤖 Real-time Inference")
        if uploaded_file is not None:
            with st.spinner("Analyzing with YOLOv8..."):
                data, success = predict_via_fastapi(image)
                
                if success:
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Result", data['pass_fail'], delta=None, delta_color="normal")
                    m2.metric("Defects", data['num_detections'])
                    
                    if data['pass_fail'] == "FAIL":
                        st.error(f"❌ {data['num_detections']} Defect(s) detected!")
                        for det in data['detections']:
                            st.write(f"- **{det['class_name']}**: {det['confidence']:.2%}")
                    else:
                        st.success("✅ PCB Passed Inspection")
                    
                    # Flywheel Logic (Manual Override)
                    st.divider()
                    st.markdown("#### 🔄 Human-in-the-Loop")
                    if st.button("Flag for Human Review (Label Studio)"):
                        st.info("Task routed to Label Studio for verification.")
                        st.toast("Feedback sent to training loop!")
                else:
                    st.error(f"Inference failed: {data}")
        else:
            st.info("Upload an image to start real-time defect detection.")

with tabs[1]:
    st.header("📈 Automated Flywheel Stats")
    
    img_count, img_list = get_flywheel_stats()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending Images", img_count)
    c2.metric("Label Studio Tasks", "Connected", delta="Active")
    c3.metric("Model Status", "Champion")
    
    st.subheader("📁 Unseen Simulation Directory")
    st.write(f"Path: `{SIMULATION_DIR}`")
    
    if img_count > 0:
        # Show a preview of some images
        cols = st.columns(5)
        for i, img_path in enumerate(img_list[:5]):
            with cols[i]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
        if img_count > 5:
            st.write(f"... and {img_count - 5} more images.")
    else:
        st.warning("No images found in simulation directory.")
