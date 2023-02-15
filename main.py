from flask import Flask, request, jsonify
from PIL import Image
import io
from classify.predict import UseModel
app = Flask(__name__)
import os
import time

model = UseModel(weights=["car_reco_model.pt"])

@app.route("/predict", methods=["POST"])
def predict():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        image_stream = io.BytesIO(im_bytes)
        
        name = time.time()
        im = im.save(f"./to_predict/{name}.jpg")
        result = model.predict(f"to_predict/{name}.jpg")

    return jsonify({'msg': 'success', 'predicted': result[0]})

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

        