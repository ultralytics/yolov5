<div align="center" markdown>
<img src="https://i.imgur.com/CH8PfN5.png"/>

# Serve YOLOv5

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
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

App deploys model (pretrained on COCO or custom one) as REST API service.

# How To Run

**For pretrained model**: just choose weights from dropdown menu and press `Run`. 

<img src="https://i.imgur.com/SEuE2jD.png" width="400"/>


**For custom weights**: 

1. Training app saves artifacts to `Team Files`. Just copy path to weights `.pt` file. 
   Training app saves results to the directory: `/yolov5_train/<training project name>/<session id>/weights`. 
   For example: `/yolov5_train/lemons_annotated/2577/weights/best.pt`

<img src="https://i.imgur.com/VkSS58q.gif"/>

2. Paste path to modal window

<img src="https://i.imgur.com/YbnwzI7.png" width="400"/>

Then

3. **Choose device (optional)**: for GPU just provide device id (`0` or `1` or ...), or type `cpu`. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="https://i.imgur.com/wKs7zw0.png"/>