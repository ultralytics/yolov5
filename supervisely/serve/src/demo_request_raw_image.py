import requests
import json
from pathlib import Path
import os
import supervisely.imaging.image as sly_image

# Start debug in Advanced Mode for Supervisely Team or run Serve app from ecosystem and after run this script from 'sly_integration/serve/src' path

APP_ADDRESS = "https://app.supervise.ly/net/dwuDUS_vp1V9UX...secret_url...J7_lJVUvfBplBZAm0="
ENDPOINT_URL = f'{APP_ADDRESS}/inference_image' # '/inference_batch' endpoint is also available.
ROOT_SOURCE_PATH = str(Path(os.getcwd()).parents[2])
IMAGE_PATH = os.path.join(ROOT_SOURCE_PATH, "data", "images", "bus.jpg")
SETTINGS = json.dumps({
    'settings': {'conf_thres': 0.4}
})


def run():
    image_np = sly_image.read(IMAGE_PATH)
    image_bytes = sly_image.write_bytes(image_np, 'jpeg')
    image_name = os.path.basename(IMAGE_PATH)
    r = requests.post(ENDPOINT_URL, files=(
        # settings can be omitted
        ('settings', (None, SETTINGS, 'text/plain')),
        ('files', (image_name, image_bytes)),
        # for multiple files add more tuples (for /inference_batch)
        # ('files', open(IMAGE_PATH, 'rb')),
    ))
    print(r.status_code)
    print(r.json())

if __name__ == '__main__':
    run()