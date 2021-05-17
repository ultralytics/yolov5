<div align="center" markdown>
<img src="https://i.imgur.com/YwSq29o.png"/>

# Train YOLOv5

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/train&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Train YOLOv5 on your custom data. All annotations will be converted to the bounding boxes automatically. Configure Train / Validation splits, model and training hyperparameters. Run on any agent (with GPU) in your team. Monitor progress, metrics, logs and other visualizations withing a single dashboard.  


Major releases:
- **May 17, 2021**: [v5.0 release](https://github.com/supervisely-ecosystem/yolov5/tree/v5.0.0): merge updates from original YOLOv5 repo (including new model architectures), split data to train/val based on datasets or tags, update UI for settings, other fixes
- **March 3, 2021**: [v4.0 release](https://github.com/supervisely-ecosystem/yolov5/tree/v4.0.9): YOLOv5 is integrated to Supervisely (train / serve / inference)

# How To Use

1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it 
3. Run app from context menu of images project
4. Open Training Dashboard (app UI) and follow instructions provided in the video below
5. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory is provided in output card in UI. 
   
   Save path is the following: ```"/yolov5_train/<input project name>/<task id>```

   For example: ```/yolov5_train/lemons-train/2712```



Watch short video for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/e47rWdgK-_M" data-video-code="e47rWdgK-_M">
    <img src="https://i.imgur.com/sJdEEkN.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>


# Screenshot

<img src="https://i.imgur.com/eiROUgb.png"/>
