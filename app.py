# app.py

import streamlit as st
from PIL import Image
from infer import predict_pil_image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# Sidebar info
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")
    st.markdown(
        """
        This demo uses a **PyTorch CNN** trained on brain MRI images  
        to classify tumors into one of the known classes.

        **Steps:**
        1. Upload an MRI image (JPG/PNG)
        2. Wait for the model to run
        3. See prediction + confidence
        """
    )
    st.markdown("---")
    st.markdown("Built by **Velpula Siva Reddy**")

st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI scan below to get a predicted tumor class.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Show the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    with st.spinner("Running model prediction..."):
        label, conf = predict_pil_image(image)

    st.success("Prediction complete ✔")

    st.markdown(
        f"""
        ### 🧾 Result  
        **Predicted Class:** `{label}`  
        **Confidence:** `{conf*100:.2f}%`
        """
    )
else:
    st.info("Please upload an MRI image to get started.")
