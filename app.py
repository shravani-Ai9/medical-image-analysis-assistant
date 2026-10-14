
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import cv2
import gdown
from PIL import Image


st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="wide"
)

# Professional dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8FC;
    }
    [data-testid="stHeader"] {
        background-color: rgba(245, 248, 252, 0.95);
    }
    h1, h2, h3 {
        color: #17365D;
    }
    [data-testid="stSidebar"] {
        background-color: #EAF0F8;
        border-right: 1px solid #D5DFEC;
    }
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCE5F0;
        padding: 16px;
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] {
        background-color: white;
        border: 1px dashed #91A9C7;
        border-radius: 12px;
        padding: 12px;
    }
    div.stButton > button, div.stDownloadButton > button {
        background-color: #17365D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1rem;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #24558A;
        color: white;
    }
    hr {
        border-color: #DCE5F0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar project information
with st.sidebar:
    st.title("🩻 Medical AI")
    st.caption("IMAGE ANALYSIS PLATFORM")
    st.divider()
    st.markdown("### Project Overview")
    st.write("Chest X-ray classification with visual explanations and downloadable reports.")
    st.markdown("### Model Information")
    st.write("**Architecture:** EfficientNetB0")
    st.write("**Input size:** 224 × 224")
    st.write("**Output classes:**")
    st.write("• BACTERIAL")
    st.write("• NORMAL")
    st.write("• VIRAL")
    st.divider()
    st.caption("Educational prototype only. Not for clinical diagnosis.")

st.title("🩻 AI-Powered Medical Image Analysis")
st.caption("Educational prototype — not for clinical diagnosis")


# Model performance and safety notice
st.info(
    "Model validation accuracy: approximately 81.11% on the current validation setup. "
    "This is a preliminary educational result, not evidence of clinical reliability. "
    "The model must not be used to diagnose or guide treatment."
)

# Dashboard overview
st.markdown("## Dashboard Overview")
st.write(
    "Analyze a chest X-ray, review model output scores, "
    "explore Grad-CAM visualizations, and export an educational report."
)

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric("AI Task", "X-ray Analysis")

with card2:
    st.metric("Model", "EfficientNetB0", help="Convolutional neural network architecture")

with card3:
    st.metric("Input Size", "224 × 224")

with card4:
    st.metric("Output Classes", len(class_names) if "class_names" in globals() else 3)



PROJECT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR, "medical_xray_efficientnet.keras"
)
CLASSES_PATH = os.path.join(PROJECT, "class_names.json")

GOOGLE_DRIVE_FILE_ID = "1qX7JPvUy1XPD8_hYHqWe7cn7uctBPicA"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model..."):
        downloaded_file = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=MODEL_PATH,
            quiet=False
        )

        if not downloaded_file or not os.path.exists(MODEL_PATH):
            st.error("Model download failed. Check Google Drive sharing permissions.")
            st.stop()

@st.cache_resource
def load_ai_model(path):
    return tf.keras.models.load_model(path)

model = load_ai_model(MODEL_PATH)



with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

def make_gradcam(image_array, class_index):
    base_model = model.layers[0]
    conv_layer = base_model.get_layer("top_conv")

    grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        feature_maps, base_output = grad_model(
            image_array, training=False
        )
        # Apply the remaining classification layers
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        score = x[:, class_index]

    gradients = tape.gradient(score, feature_maps)
    if gradients is None:
        raise ValueError("Unable to calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    original = np.uint8(image_array[0])
    overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

    return heatmap, overlay

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=400)

    from io import BytesIO
    xray_buffer = BytesIO()
    image.save(xray_buffer, format="PNG")
    st.session_state["xray_image_for_pdf"] = xray_buffer.getvalue()

    if st.button("Analyze Image"):
        resized = image.resize((224, 224))
        array = np.array(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)

        with st.spinner("Analyzing image..."):
            probabilities = model.predict(batch, verbose=0)[0]
            index = int(np.argmax(probabilities))

            st.session_state["predicted_class"] = class_names[index]
            st.session_state["probabilities"] = probabilities.tolist()

            # Store prediction in session history
            if "analysis_history" not in st.session_state:
                st.session_state["analysis_history"] = []

            st.session_state["analysis_history"].append({
                "Analysis": len(st.session_state["analysis_history"]) + 1,
                "Predicted Class": class_names[index],
                "Top Model Score (%)": round(float(probabilities[index]) * 100, 2)
            })

        st.subheader("Model Output")
        st.write(f"**Predicted class:** {class_names[index]}")
        st.caption("Model prediction only — not a medical diagnosis.")

        st.subheader("Class Scores")
        for name, score in zip(class_names, probabilities):
            st.write(f"{name}: {score * 100:.2f}%")
            st.progress(float(score))

        # Prediction score chart
        import pandas as pd

        st.subheader("Prediction Score Comparison")
        chart_data = pd.DataFrame({
            "Class": class_names,
            "Score (%)": [float(score) * 100 for score in probabilities]
        }).sort_values("Score (%)", ascending=False)

        st.bar_chart(
            chart_data.set_index("Class"),
            y="Score (%)"
        )
        st.caption(
            "Scores represent model outputs, not confirmed disease probabilities "
            "or a medical diagnosis."
        )

        try:
            heatmap, overlay = make_gradcam(batch, index)

            from PIL import Image as PILImage
            overlay_buffer = BytesIO()
            PILImage.fromarray(overlay).save(overlay_buffer, format="PNG")
            st.session_state["gradcam_image_for_pdf"] = overlay_buffer.getvalue()

            st.subheader("Grad-CAM Explainability")
            st.caption(
                "Highlighted regions show image areas influencing the model, "
                "not confirmed medical findings."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(heatmap, caption="Grad-CAM Heatmap",
                         clamp=True, use_container_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Overlay",
                         clamp=True, use_container_width=True)

        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")

# Analysis History
if st.session_state.get("analysis_history"):
    st.divider()
    st.subheader("📋 Analysis History")
    import pandas as pd
    history_df = pd.DataFrame(st.session_state["analysis_history"])
    st.dataframe(history_df, use_container_width=True, hide_index=True)

from report_generator import generate_report

# Educational report download
if uploaded_file is not None and "probabilities" in st.session_state:
    st.divider()
    st.subheader("📄 AI-Generated Explanatory Report")

    report_text = generate_report(
        st.session_state["predicted_class"],
        st.session_state["probabilities"]
    )

    st.markdown(report_text)

    st.download_button(
        label="Download Analysis Report",
        data=report_text,
        file_name="AI_Chest_Xray_Analysis_Report.md",
        mime="text/markdown"
    )

from pdf_report_generator import create_pdf_report

# PDF report download
if uploaded_file is not None and "probabilities" in st.session_state:
    st.divider()
    st.subheader("📑 Download Professional PDF Report")

    pdf_bytes = create_pdf_report(
        predicted_class=st.session_state["predicted_class"],
        probabilities=st.session_state["probabilities"],
        class_names=class_names,
        xray_image=st.session_state.get("xray_image_for_pdf"),
        gradcam_image=st.session_state.get("gradcam_image_for_pdf")
    )

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name="AI_Chest_Xray_Analysis_Report.pdf",
        mime="application/pdf",
        key="download_pdf_report"
    )


# Professional project footer
st.divider()
st.markdown(
    """
    <div style="text-align:center; padding:12px; color:#64748b;">
        <strong>AI-Powered Multi-Modal Medical Image Analysis &amp; Report Assistant</strong>
        <br>
        <span>Educational prototype • EfficientNetB0 • Grad-CAM Explainability</span>
        <br>
        <small>Not intended for medical diagnosis or treatment decisions.</small>
    </div>
    """,
    unsafe_allow_html=True
)
