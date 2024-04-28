import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory

# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
# from werkzeug.utils import secure_filename
# import os
# from wtforms.validators import InputRequired


app = Flask(__name__)

streaming_active = False
output_folder = "videos"
video_writer = None
# class UploadFileForm(FlaskForm):
#     file = FileField("File", validators=[InputRequired()])
#     submit = SubmitField("Upload File")


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/home', methods=['GET',"POST"])
# def home():
#     form = UploadFileForm()
#     if form.validate_on_submit():
#         file = form.file.data # First grab the file
#         file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
#         return "File has been uploaded."
#     return render_template('index.html', form=form)


@app.route("/start_stream")
def start_stream():
    global streaming_active
    global out
    if not streaming_active:
        streaming_active = True
        start_recording()

        return jsonify({"status": "success", "message": "Streaming started and recording initiated"})
    else:
        return jsonify({"status": "error", "message": "Streaming is already active"})


@app.route("/stop_stream")
def stop_stream():
    global streaming_active
    if streaming_active:
        streaming_active = False
        stop_recording()

        return jsonify({"status": "success", "message": "Streaming stopped and recording saved"})
    else:
        return jsonify({"status": "error", "message": "Streaming is not active"})


def start_recording():
    global video_writer

    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as needed
    frame_size = (640, 480)  # Adjust the frame size as needed
    # video_writer = cv2.VideoWriter(filename, fourcc, 10.0, frame_size)
    video_writer = cv2.VideoWriter(os.path.join(output_folder, filename), fourcc, 40.0, frame_size)


def stop_recording():
    global video_writer

    if video_writer is not None:
        video_writer.release()
        video_writer = None


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/videos")
def list_videos():
    videos = [video for video in os.listdir("videos") if video.endswith(".mp4")]
    return jsonify(videos)


@app.route("/video/<filename>")
def stream_video(filename):
    return send_from_directory("videos", filename)


@app.route("/detection/<filename>")
# def detection(filename):
#     # Placeholder for detection logic
#     print(f"Detection started for {filename}")
#     return jsonify({'status': 'Detection started for ' + filename})
def detection(filename):
    try:
        print(f"filename################={filename}")
        # Construct the command string
        command = f"python3 detect.py --source ./videos/{filename}"

        # Execute the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            # Handle errors if the command failed
            return jsonify({"status": "Error", "message": stderr.decode()}), 500

        # Return success response
        return jsonify({"status": "Detection Done for " + filename})
    except Exception as e:
        # Handle any exceptions
        return jsonify({"status": "Error", "message": str(e)}), 500


def generate_frames():
    global video_writer
    folder_path = "/tmp/camera_save_tutorial"
    while streaming_active:
        image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
        if image_files:
            try:
                latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                image_path = os.path.join(folder_path, latest_image)

                frame = cv2.imread(image_path)
                if frame is None:
                    raise FileNotFoundError("Empty image file or format not supported")

                _, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()

                # Write the frame to the video file if recording is active
                if video_writer is not None:
                    video_writer.write(cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), 1))

                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                # time.sleep(0.05)

            except Exception as e:
                print(f"Error processing image: {e}")
                continue

        else:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + b"\r\n")


if __name__ == "__main__":
    app.run(debug=True)
