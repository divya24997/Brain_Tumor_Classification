import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import h5py

# ----------------------------
# Build EfficientNetB0
# ----------------------------
num_classes = 4

def build_effnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def load_model_h5(model, filename):
    with h5py.File(filename, 'r') as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = torch.tensor(f[key][()])
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = build_effnet()
model = load_model_h5(model, "efficientnetb0.h5")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image, model):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
    return pred_class, probs

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# ======= CUSTOM CSS (Blue Background + Center Alignment + Fit Screen) =======
st.markdown(
    """
    <style>
    /* Blue Background */
    .stApp {
        background: linear-gradient(135deg, #0b3d91, #1e63d1);
    }

    /* Center all page content */
    .block-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        max-width: 900px;  /* limit width */
        padding-top: 40px;
        padding-bottom: 40px;
    }

    /* Title */
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #ffffff;
        margin-bottom: 25px;
    }

    /* Box */
    .box {
        background-color: rgba(255, 255, 255, 0.12);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.20);
        width: 100%;
        text-align: center;
        color: white;
    }

    /* Center button */
    .button-center {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }

    /* Button style */
    .stButton>button {
        color: black;
        background-color: #ffffff;
        font-weight: bold;
    }

    /* File uploader text color */
    .stFileUploader {
        color: white;
    }

    /* Make all output text white */
    .stText, .stMarkdown, .stSuccess, .stInfo {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======= PAGE NAVIGATION =======
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- HOME PAGE ----------
if st.session_state.page == "home":
    st.markdown("<div class='title'>Brain Tumor Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-based MRI tumor detection using EfficientNetB0</div>", unsafe_allow_html=True)

    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.write("### Project Description")
    st.write("""
    This project uses a deep learning model to classify brain MRI images into 4 categories:
    - **Glioma**
    - **Meningioma**
    - **No Tumor**
    - **Pituitary Tumor**

    Click **Predict** to upload an MRI image and get the tumor type with confidence scores.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='button-center'>", unsafe_allow_html=True)
    if st.button("Predict"):
        st.session_state.page = "predict"
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICT PAGE ----------
if st.session_state.page == "predict":
    st.markdown("<div class='title'>Upload MRI Image</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Get tumor prediction and explanation</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # ---- shrink image ----
        st.image(image, caption="Uploaded MRI", width=350)

        pred_class, probs = predict(image, model)

        st.write("## Prediction")
        
        # ---- tumor type in BLACK ----
        st.markdown(
            f"<h2 style='color:black;'>Tumor Type: <b>{class_names[pred_class]}</b></h2>",
            unsafe_allow_html=True
        )

        st.write("## Confidence Scores")
        for i, p in enumerate(probs):
            st.write(f"{class_names[i]}: {p*100:.2f}%")

        st.write("---")
        st.write("## Description")
        if class_names[pred_class] == "glioma":
            st.write("Glioma is a type of tumor that occurs in the brain and spinal cord.")
        elif class_names[pred_class] == "meningioma":
            st.write("Meningioma is a tumor that forms on membranes covering the brain and spinal cord.")
        elif class_names[pred_class] == "pituitary":
            st.write("Pituitary tumor is a growth in the pituitary gland which controls hormones.")
        else:
            st.write("No tumor detected. This indicates a normal MRI scan.")

    if st.button("Back to Home"):
        st.session_state.page = "home"
