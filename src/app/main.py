import streamlit as st
import PIL.Image
import requests
import numpy as np
import os
import time
from pathlib import Path
import subprocess

st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
SIMULATION_DIR = Path("data/raw/unseen_simulation")

# --- Helper Functions ---
def draw_boxes(image, detections):
    """Draws bounding boxes and labels on the image."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections:
        box = det['bbox_xyxy']
        label = f"{det['class_name']} {det['confidence']:.2%}"
        
        # Draw rectangle
        draw.rectangle(box, outline="#ff4b4b", width=4)
        
        # Draw label background
        text_size = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([box[0], box[1] - 25, box[0] + (text_size[2]-text_size[0]) + 10, box[1]], fill="#ff4b4b")
        
        # Draw text
        draw.text((box[0] + 5, box[1] - 25), label, fill="white", font=font)
    
    return image
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
    
    if st.button("🚀 Trigger Active Learning Loop", use_container_width=True):
        with st.status("Running Batch Inference...", expanded=True) as status:
            st.write("Scanning `data/raw/unseen_simulation`...")
            # We run the script as a subprocess to simulate a real MLOps trigger
            result = subprocess.run(["python", "src/utils/batch_inference.py"], capture_output=True, text=True)
            st.code(result.stdout)
            status.update(label="Active Learning Cycle Complete!", state="complete", expanded=False)
        st.balloons()

# --- MAIN UI ---
st.title("🔍 PCB Defect Detection System")
st.markdown("### MLOps Active Learning Pipeline")

tabs = st.tabs(["🎮 Interactive Sandbox", "📊 Dashboard"])

with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Capture")
        uploaded_file = st.file_uploader("Drop a PCB image here...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded PCB", use_container_width=True)

    with col2:
        st.header("🤖 Detection Results")
        if uploaded_file is not None:
            with st.spinner("Analyzing with YOLOv8..."):
                data, success = predict_via_fastapi(image)
                
                if success:
                    # Draw boxes on a copy of the image
                    annotated_image = draw_boxes(image.copy(), data['detections'])
                    st.image(annotated_image, caption="Inference Result", use_container_width=True)

                    # Metrics
                    m1, m2 = st.columns(2)
                    result_color = "inverse" if data['pass_fail'] == "FAIL" else "normal"
                    m1.metric("Status", data['pass_fail'], delta=None, delta_color=result_color)
                    m2.metric("Defects Found", data['num_detections'])
                    
                    if data['pass_fail'] == "FAIL":
                        st.error(f"⚠️ {data['num_detections']} Defect(s) detected!")
                    else:
                        st.success("✅ PCB Passed Inspection")
                    
                    # Details
                    with st.expander("View Detection Details"):
                        st.json(data['detections'])
        else:
            st.info("Upload an image to start real-time defect detection.")

with tabs[1]:
    st.header("📈 Active Learning Stats")
    
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
