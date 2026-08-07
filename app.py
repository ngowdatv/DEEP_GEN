import streamlit as st
import pandas as pd
from utils.helper import detect_dataset_type

st.set_page_config(
    page_title="Deep Generative Framework",
    layout="wide"
)

st.title("🧠 Deep Generative Framework for Synthetic Data Generation")

st.write(
    "Upload an Image, CSV, or Text dataset to generate synthetic data."
)

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["jpg", "jpeg", "png", "csv", "txt"]
)

if uploaded_file is not None:

    dataset_type = detect_dataset_type(uploaded_file.name)

    st.success(f"Dataset Type: {dataset_type.upper()}")

    if dataset_type == "image":
        st.image(uploaded_file, caption="Uploaded Image")

    elif dataset_type == "csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

    elif dataset_type == "text":
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Text Preview", text, height=300)

    else:
        st.error("Unsupported dataset.")