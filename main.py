from flask import Flask, request, jsonify
from PIL import Image
import io
from classify.predict import UseModel
app = Flask(__name__)
import os

model = UseModel(weights=["car_reco_model.pt"])

@app.route("/predict", methods=["POST"])
def predict():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        im = im.save("./to_predict/im.jpg")
        result, possible = model.predict("to_predict/im.jpg")

    return jsonify({'msg': 'success', 'predicted': result[0], 'possible': possible[0]})

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

        