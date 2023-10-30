from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('input.html')

@app.route("/extract_license_plate", methods=['POST'])
def extract_license_plate():
    if not request.method == "POST":
        return
    
    image = request.files['image']
    image.save(secure_filename(image.filename))

    # Call the detect_3.py script and capture its output
    import subprocess
    cmd = [
        'python',
        'C:\\Users\\manik\\Downloads\\yolov5s\\yolov5\\detect_3.py',
        '--weights',
        'C:\\Users\\manik\\Downloads\\yolov5s\\yolov5\\runs\\train\\exp\\weights\\last.pt',
        '--img', '640',
        '--conf', '0.25',
        '--source', secure_filename(image.filename)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Extract the license plate text from the result
    extracted_text = result.stdout

    return render_template('result.html',extracted_text=extracted_text)

if __name__ == '__main__':
    app.run()
