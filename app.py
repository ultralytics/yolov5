import base64
import json
import requests
import time
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from utils_plot import plot_image_human_detection, plot_image_mask_detection


def encode_image(image):
    """Encode an image to base64 encoded bytes.
    Args:
        image: PIL.PngImagePlugin.PngImageFile
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def image_recognize():
    st.title("Test endpoint")

    st.sidebar.info(
        "**Note**: For\n"
        "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
        "change **http** to **https** in the API URL.")

    mode = st.selectbox("Select detection type.", ["Human detection", "Mask detection"])
    url = st.text_input("Input API URL.", "http://127.0.0.1:5000/")
    token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

        start_time = time.time()
        response_json = recognize(image, url, token)
        latency = time.time() - start_time

        st.subheader("API Response")
        st.write(f"**Est. latency = `{latency:.3f} s`**")
        st.text(json.dumps(response_json, indent=2))

        st.subheader("Output Image with Bounding Boxes")
        if mode == "Human detection":
            output_image = plot_image_human_detection(np.asarray(image.convert("RGB")), response_json)
        else:
            output_image = plot_image_mask_detection(np.asarray(image.convert("RGB")), response_json)
        st.image(Image.fromarray(output_image), use_column_width=True)


if __name__ == "__main__":
    image_recognize()
