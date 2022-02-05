import streamlit as st
import cv2
import numpy as np


def infer_detection(weights_path: str, im: np.ndarray, confidence: float):
    from Repsycle.inference.detection.yolov5_utils import Yolov5
    model = Yolov5(
        weights=weights_path,
        preprocessing_file_path=None,
        conf_thresh=confidence,
        labels={'0': {'name': 'zero', 'color': [255, 0, 0]},
                '1': {'name': 'un', 'color': [0, 255, 0]},
                '2': {'name': 'deux', 'color': [0, 0, 255]}},
    )
    out = model.infer(im)
    print(out.output_instances)
    im_out = model.draw(im, out, print_confidence=True, draw_center=True)
    cv2.imwrite('/tmp/in.png', im)
    cv2.imwrite('/tmp/out.png', im_out)
    return im_out


if __name__ == '__main__':
    st.title('Test weights file on image')
    weights_path = st.text_input("Weights file path")
    upload_image = st.file_uploader("Image")
    confidence = st.slider('Confidence threshold', 0, 100, 50)
    if st.button('Go') and weights_path and upload_image and confidence:
        # Image
        im = cv2.imdecode(np.asarray(bytearray(upload_image.read()), dtype=np.uint8), 1)
        im_out = infer_detection(weights_path, im, confidence / 100.)
        st.image(im_out, channels='BGR')
