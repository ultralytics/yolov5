<div align="center" markdown>
<img src="https://i.imgur.com/1qXIdqs.png"/>

# Serve YOLOv5

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Result-JSON-Format">For Developers</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App deploys YOLO v5 model (pretrained on COCO or custom one) as REST API service. Serve app is the simplest way how any model can be integrated into Supervisely. Once model is deployed, user gets the following benefits:

1. Use out of the box apps for inference
   - model can be used directly in [labeling interface](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) (images, videos)
   - model can be applied to [images project or dataset](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset)
2. Apps from Supervisely Ecosystem can use NN predictions: for visualization, for analysis, performance evaluation, etc ...
3. Communicate with NN in custom python script (see section <a href="#For-developers">for developers</a>)
4. App illustrates how to use NN weights. For example: you can train model in Supervisely, download its weights and use them the way you want.

Watch usage demo:

<a data-key="sly-embeded-video-link" href="https://youtu.be/cMBhn1Erluk" data-video-code="cMBhn1Erluk">
    <img src="https://i.imgur.com/UlEMeem.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>


# How To Run

**For pretrained model**: just choose weights from dropdown menu and press `Run`. 

<img src="https://i.imgur.com/SEuE2jD.png" width="400"/>


**For custom weights**: 

1. Training app saves artifacts to `Team Files`. Just copy path to weights `.pt` file. 
   Training app saves results to the directory: `/yolov5_train/<training project name>/<session id>/weights`. 
   For example: `/yolov5_train/lemons_annotated/2577/weights/best.pt`

<img src="https://i.imgur.com/VkSS58q.gif" width="800"/>

2. Paste path to modal window

<img src="https://i.imgur.com/YbnwzI7.png" width="400"/>

Then

3. Choose device (optional): for GPU just provide device id (`0` or `1` or ...), or type `cpu`. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="https://i.imgur.com/wKs7zw0.png" width="800"/>


# For developers

This python example illustrates available methods of the deployed model. Now you can integrate network predictions to your python script. This is the way how other Supervisely Apps can communicate with NNs. And also you can use serving app as an example - how to use download NN weights outside Supervisely.

To implement serving app developer has just to define four methods:
- function [`get_session_info`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/sly_serve.py#L50) information about deployed model - just returns python dictionary with any useful information
- function [`construct_model_meta`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L16) - returns model output classes and tags in [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)
- function [`load_model`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L37) - how to load model to the device (cpu or/and gpu) - [link](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/sly_serve.py#L165)
- function [`inference`](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/serve/src/nn_utils.py#L62)  - how to apply model to the image and how to convert predictions to [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)


## Python Example: how to communicate with deployed model 
```python
import json
import yaml
import numpy as np
import supervisely_lib as sly


def visualize(img: np.ndarray, ann: sly.Annotation, name, roi: sly.Rectangle = None):
    vis = img.copy()
    if roi is not None:
        roi.draw_contour(vis, color=[255, 0, 0], thickness=3)
    ann.draw_contour(vis, thickness=3)
    sly.image.write(f"./images/{name}", vis)


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 2723

    # get information about model
    info = api.task.send_request(task_id, "get_session_info", data={})
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get model output classes and tags
    meta_json = api.task.send_request(task_id, "get_output_classes_and_tags", data={})
    model_meta = sly.ProjectMeta.from_json(meta_json)
    print("Model produces following classes and tags")
    print(model_meta)

    # get model inference settings (optional)
    resp = api.task.send_request(task_id, "get_custom_inference_settings", data={})
    settings_yaml = resp["settings"]
    settings = yaml.safe_load(settings_yaml)
    # you can change this default settings and pass them to any inference method
    print("Model inference settings:")
    print(json.dumps(settings, indent=4))

    # inference for url
    image_url = "https://i.imgur.com/tEkCb69.jpg"

    # download image for further debug visualizations
    save_path = f"./images/{sly.fs.get_file_name_with_ext(image_url)}"
    sly.fs.ensure_base_path(save_path)  # create directories if needed
    sly.fs.download(image_url, save_path)
    img = sly.image.read(save_path)  # RGB

    # apply model to image URl (full image)
    # you can pass 'settings' dictionary to any inference method
    # every model defines custom inference settings
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "settings": settings,
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "01_prediction_url.jpg")

    # apply model to image URL (only ROI - region of interest)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width/2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "02_prediction_url_roi.jpg", roi)

    # apply model to image id (full image)
    image_id = 770730
    ann_json = api.task.send_request(task_id, "inference_image_id", data={"image_id": image_id})
    ann = sly.Annotation.from_json(ann_json, model_meta)
    img = api.image.download_np(image_id)
    visualize(img, ann, "03_prediction_id.jpg")

    # apply model to image id (only ROI - region of interest)
    image_id = 770730
    img = api.image.download_np(image_id)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width / 2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "04_prediction_id_roi.jpg", roi)

    # apply model to several images (using id)
    batch_ids = [770730, 770727, 770729, 770720]
    resp = api.task.send_request(task_id, "inference_batch_ids", data={"batch_ids": batch_ids})
    for ind, (image_id, ann_json) in enumerate(zip(batch_ids, resp)):
        ann = sly.Annotation.from_json(ann_json, model_meta)
        img = api.image.download_np(image_id)
        visualize(img, ann, f"05_prediction_batch_{ind:03d}_{image_id}.jpg")


if __name__ == "__main__":
    main()
```

## Example Output

Information about deployed model:

```json
{
    "app": "YOLOv5 serve",
    "weights": "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt",
    "device": "cuda:0",
    "half": "True",
    "input_size": 640,
    "session_id": "2723",
    "classes_count": 80,
    "tags_count": 1
}
```

Model produces following classes and tags:
```
ProjectMeta:
Object Classes
+----------------+-----------+----------------+--------+
|      Name      |   Shape   |     Color      | Hotkey |
+----------------+-----------+----------------+--------+
|     person     | Rectangle | [36, 15, 138]  |        |
|    bicycle     | Rectangle | [113, 138, 15] |        |
|      car       | Rectangle | [138, 15, 53]  |        |
|   motorcycle   | Rectangle | [15, 138, 101] |        |
|    airplane    | Rectangle | [138, 75, 15]  |        |
|      bus       | Rectangle | [20, 138, 15]  |        |
|     train      | Rectangle | [125, 15, 138] |        |
|     truck      | Rectangle | [15, 73, 138]  |        |
|      boat      | Rectangle | [15, 127, 138] |        |
| traffic light  | Rectangle | [138, 15, 102] |        |
|  fire hydrant  | Rectangle | [15, 138, 55]  |        |
|   stop sign    | Rectangle | [138, 24, 15]  |        |
| parking meter  | Rectangle | [65, 138, 15]  |        |
|     bench      | Rectangle | [79, 15, 138]  |        |
|      bird      | Rectangle | [138, 116, 15] |        |
|      cat       | Rectangle | [15, 37, 138]  |        |
|      dog       | Rectangle | [15, 98, 138]  |        |
|     horse      | Rectangle | [138, 48, 15]  |        |
|     sheep      | Rectangle | [138, 15, 79]  |        |
|      cow       | Rectangle | [138, 15, 127] |        |
|    elephant    | Rectangle | [15, 138, 124] |        |
|      bear      | Rectangle | [89, 138, 15]  |        |
|     zebra      | Rectangle | [135, 138, 15] |        |
|    giraffe     | Rectangle | [15, 138, 77]  |        |
|    backpack    | Rectangle | [138, 15, 27]  |        |
|    umbrella    | Rectangle | [101, 15, 138] |        |
|    handbag     | Rectangle | [17, 15, 138]  |        |
|      tie       | Rectangle | [15, 138, 33]  |        |
|    suitcase    | Rectangle | [40, 138, 15]  |        |
|    frisbee     | Rectangle | [138, 96, 15]  |        |
|      skis      | Rectangle | [60, 15, 138]  |        |
|   snowboard    | Rectangle | [15, 55, 138]  |        |
|  sports ball   | Rectangle | [15, 114, 138] |        |
|      kite      | Rectangle | [138, 15, 66]  |        |
|  baseball bat  | Rectangle | [52, 138, 15]  |        |
| baseball glove | Rectangle | [138, 129, 15] |        |
|   skateboard   | Rectangle | [101, 138, 15] |        |
|   surfboard    | Rectangle | [138, 36, 15]  |        |
| tennis racket  | Rectangle | [138, 61, 15]  |        |
|     bottle     | Rectangle | [15, 138, 89]  |        |
|   wine glass   | Rectangle | [77, 138, 15]  |        |
|      cup       | Rectangle | [138, 15, 115] |        |
|      fork      | Rectangle | [15, 138, 21]  |        |
|     knife      | Rectangle | [48, 15, 138]  |        |
|     spoon      | Rectangle | [138, 15, 41]  |        |
|      bowl      | Rectangle | [15, 25, 138]  |        |
|     banana     | Rectangle | [138, 106, 15] |        |
|     apple      | Rectangle | [137, 15, 138] |        |
|    sandwich    | Rectangle | [15, 86, 138]  |        |
|     orange     | Rectangle | [114, 15, 138] |        |
|    broccoli    | Rectangle | [90, 15, 138]  |        |
|     carrot     | Rectangle | [15, 138, 136] |        |
|    hot dog     | Rectangle | [15, 138, 67]  |        |
|     pizza      | Rectangle | [138, 85, 15]  |        |
|     donut      | Rectangle | [138, 15, 17]  |        |
|      cake      | Rectangle | [15, 46, 138]  |        |
|     chair      | Rectangle | [124, 138, 15] |        |
|     couch      | Rectangle | [138, 15, 88]  |        |
|  potted plant  | Rectangle | [30, 138, 15]  |        |
|      bed       | Rectangle | [15, 138, 44]  |        |
|  dining table  | Rectangle | [69, 15, 138]  |        |
|     toilet     | Rectangle | [15, 138, 114] |        |
|       tv       | Rectangle | [27, 15, 138]  |        |
|     laptop     | Rectangle | [138, 15, 72]  |        |
|     mouse      | Rectangle | [15, 106, 138] |        |
|     remote     | Rectangle | [15, 133, 138] |        |
|    keyboard    | Rectangle | [15, 63, 138]  |        |
|   cell phone   | Rectangle | [138, 68, 15]  |        |
|   microwave    | Rectangle | [138, 15, 34]  |        |
|      oven      | Rectangle | [95, 138, 15]  |        |
|    toaster     | Rectangle | [15, 121, 138] |        |
|      sink      | Rectangle | [15, 92, 138]  |        |
|  refrigerator  | Rectangle | [58, 138, 15]  |        |
|      book      | Rectangle | [138, 15, 95]  |        |
|     clock      | Rectangle | [138, 55, 15]  |        |
|      vase      | Rectangle | [15, 79, 138]  |        |
|    scissors    | Rectangle | [15, 19, 138]  |        |
|   teddy bear   | Rectangle | [138, 15, 47]  |        |
|   hair drier   | Rectangle | [15, 138, 27]  |        |
|   toothbrush   | Rectangle | [15, 138, 83]  |        |
+----------------+-----------+----------------+--------+
Tags
+------------+------------+-----------------+--------+---------------+--------------------+
|    Name    | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
+------------+------------+-----------------+--------+---------------+--------------------+
| confidence | any_number |       None      |        |      all      |         []         |
+------------+------------+-----------------+--------+---------------+--------------------+
```

Model inference settings:
```json
{
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "augment": false,
    "debug_visualization": false
}
```

Prediction for image URL (full image):

Image URL  |  01_prediction_url.jpg
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/tEkCb69.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/9OOoXn3.jpg" style="max-height: 300px; width: auto;"/>

Prediction for image URL (ROI - red rectangle):

Image URL + ROI  |  02_prediction_url_roi.jpg
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/tEkCb69.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/iSKS17L.jpg" style="max-height: 300px; width: auto;"/>


Prediction for image id (full image):

03_input_id.jpg  |  03_prediction_id.jpg
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/RQDrH4B.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/yYujbI0.jpg" style="max-height: 300px; width: auto;"/>

Prediction for image id (ROI - red rectangle):

04_input_id_roi.jpg  |  04_prediction_id_roi.jpg
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/2XlEZQK.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/1U7413M.jpg" style="max-height: 300px; width: auto;"/>

Prediction for batch of images ids:

Image ID  |  Prediction
:-------------------------:|:-----------------------------------:
<img src="https://i.imgur.com/4Lh9tAm.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/emsah1q.jpg" style="max-height: 300px; width: auto;"/>
<img src="https://i.imgur.com/UqiV5Ka.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/GhoKKCl.jpg" style="max-height: 300px; width: auto;"/>
<img src="https://i.imgur.com/8GjoNDH.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/yzinXD6.jpg" style="max-height: 300px; width: auto;"/>
<img src="https://i.imgur.com/xOydF3B.jpg" style="max-height: 300px; width: auto;"/>  |  <img src="https://i.imgur.com/YFNmIPY.jpg" style="max-height: 300px; width: auto;"/>
