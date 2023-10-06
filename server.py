from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import tempfile
import os
import cgi
import time
import subprocess
from detect import run

# Define the request handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Handle GET requests
    def do_GET(self):
        self.send_response(200)  # Send 200 OK status code
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Send the HTML form as the response body
        form_html = '''
        <html>
        <body>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit" value="Upload" />
        </form>
        </body>
        </html>
        '''
        self.wfile.write(form_html.encode('utf-8'))

    # Handle POST requests
    def do_POST(self):
        content_type = self.headers['Content-Type']

        reqdir = "/app/tmp/" + str(time.time())+"/"
        os.makedirs(reqdir, exist_ok=True)

        # Check if the content type is multipart/form-data
        if content_type.startswith('multipart/form-data'):
            # Parse the form data
            form_data = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            # Get the uploaded file field
            file_field = form_data['file']

            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(dir="/app/tmp", delete=False) as tmp_file:
                # Save the file data to the temporary file
                tmp_file.write(file_field.file.read())

                # Get the temporary file path
                tmp_file_path = tmp_file.name

                # Extract the filename from the uploaded file field
                filename = os.path.basename(file_field.filename)

                # Move the temporary file to the new filename
                new_filename = reqdir + filename
                os.rename(tmp_file_path, new_filename)

            run(
                weights="/app/weights/best.pt",
                device="cpu",
                source=new_filename,
                project=reqdir,
                save_txt=True,
                save_conf=True,
            )

            if not os.path.exists(reqdir + "exp/result.txt"):
                self.send_response(200)  # Send 200 OK status code
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Nothing found', 'result': ""}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
                subprocess.call(["rm", "-rf", reqdir])
                return

            with open(reqdir + "exp/result.txt", 'r') as file:
                result = file.read()
            response = {'message': 'File processed successfully', 'result': result}

            subprocess.call(["rm", "-rf", reqdir])

            self.send_response(200)  # Send 200 OK status code
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

# Create an HTTP server with the request handler
server_address = ('', 8700)  # Listen on all available interfaces, port 8700
httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

# Start the server
print('Server running on port 8700')
httpd.serve_forever()
