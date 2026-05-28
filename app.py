import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import time
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐾 Animal Detection App",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1a5276, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (Cached)
# ============================================================
@st.cache_resource
def load_model(model_path):
    """Load the YOLO model (cached for performance)."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================
# DETECTION FUNCTION
# ============================================================
def detect_objects(model, image, confidence):
    """Run YOLO detection on the given image."""
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )
    return results

def draw_detections(image, results, model):
    """Draw bounding boxes and labels on the image."""
    img_with_boxes = image.copy()
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            colors = [
                (46, 204, 113), (52, 152, 219), (231, 76, 60),
                (241, 196, 15), (155, 89, 182)
            ]
            color = colors[cls_id % len(colors)]

            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)

            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                img_with_boxes,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                img_with_boxes, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

            detections.append({
                "Class": cls_name,
                "Confidence": f"{conf:.2%}",
                "X1": x1, "Y1": y1,
                "X2": x2, "Y2": y2
            })

    return img_with_boxes, detections

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<h1 class="main-header">🐾 Animal Detection using YOLO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to detect animals using a trained YOLOv8 model</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input("Model Path", value="best.pt", help="Path to your trained best.pt model file")
        confidence = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
        
        st.divider()
        st.header("ℹ️ Model Info")
        if os.path.exists(model_path):
            st.success(f"✅ Model found: {model_path}")
        else:
            st.error(f"❌ Model not found: {model_path}")
            st.info("Place best.pt in the same folder as app.py")

        st.divider()
        st.header("📖 About")
        st.markdown("""
        This app uses **YOLOv8** trained on an Animal Detection dataset.
        
        **Detected classes:**
        - 🐀 Rat
        - 🐀🐱🐶 Rat_Cat_Dog
        - 🐱 Cat
        - 🐶 Dog
        """)

    model = load_model(model_path)
    if model is None:
        st.warning("⚠️ Please provide a valid model path to continue.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

        if uploaded_file is not None:
            # FIX: Convert image to RGB to remove alpha channel (4 channels) from PNGs
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Original Image", use_container_width=True)

    with col2:
        st.header("🔍 Detection Results")
        if uploaded_file is not None:
            img_array = np.array(image)
            # FIX: Since we forced RGB above, we can safely convert directly to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            with st.spinner("🔄 Detecting animals..."):
                start_time = time.time()
                results = detect_objects(model, img_bgr, confidence)
                inference_time = time.time() - start_time

            img_with_boxes, detections = draw_detections(img_bgr, results, model)
            img_display = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

            st.image(img_display, caption=f"🎯 Detection Results ({len(detections)} animals found)", use_container_width=True)
            st.info(f"⚡ Inference Time: {inference_time:.3f} seconds")

            if detections:
                st.header("📊 Detection Details")
                det_count = len(detections)
                classes_found = list(set(d["Class"] for d in detections))
                avg_conf = np.mean([float(d["Confidence"].strip('%')) / 100 for d in detections])

                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{det_count}</div><div class="metric-label">Animals Detected</div></div>', unsafe_allow_html=True)
                with metric_cols[1]:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(classes_found)}</div><div class="metric-label">Unique Species</div></div>', unsafe_allow_html=True)
                with metric_cols[2]:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1%}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                df = pd.DataFrame(detections)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.subheader("📈 Detections per Class")
                class_counts = df["Class"].value_counts()
                st.bar_chart(class_counts)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Results as CSV", data=csv, file_name="detection_results.csv", mime="text/csv")
            else:
                st.warning("⚠️ No animals detected in this image.")
                st.info("💡 Try lowering the confidence threshold in the sidebar.")

    st.divider()
    st.markdown('<div style="text-align: center; color: #7f8c8d; padding: 1rem;">🐾 Animal Detection App | Built with YOLOv8 + Streamlit | AI Lab 8.0</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
