This guide explains how to use Supervisely with YOLOv5.

# Table of Contents

1. [About Supervisely](#About-Supervisely)
2. [Prerequisites](#Prerequisites)
3. [YOLOv5 Apps Collection](#YOLOv5-Apps-Collection)
8. [For developers](#For-developers)
9. [Contact & Questions & Suggestions](#Contact-&-Questions-&-Suggestions)

# About Supervisely

You can think of Supervisely as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as smooth and fast as possible. 

More concretely, Supervisely includes the following functionality:
 - Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
 - Data visualization and quality control
 - State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
 - Interactive tools for model performance analysis
 - Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
 - Synthetic data generation tools
 - Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the GPU related computations will be performed on the connected machine(s).


# Prerequisites
You should connect computer with GPU to your Supervisely account. If you already have Supervisely Agent running on your computer, you can skip this step.

 Several tools have to be installed on your computer:

- Nvidia drives + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

Once your computer is ready just add agent to your team and execute automatically generated running command in terminal. Watch how-to video:

<a data-key="sly-embeded-video-link" href="https://youtu.be/aDqQiYycqyk" data-video-code="aDqQiYycqyk">
    <img src="https://i.imgur.com/X9NTc5X.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:50%;">
</a>


# 🎉 YOLO v5 Apps Collection

YOLOv5 is one of the best available detectors. And we are proud to announce its full integrtion into [Supervisely Ecosystem](https://ecosystem.supervise.ly/). To learn more about how to use every app, please go to app's readme page (links are provided). Just add an interested app to your team to start using it.

<img src="https://i.imgur.com/az5sqvk.png"/>

 YOLOv5 Collection consists of the following apps: 

1. [Train YOLOv5](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fyolov5%252Fsupervisely%252Ftrain) - start training on your custom data. Just run app from the context menu of your project, choose classes of interest, train/val splits, configure training metaparameters and augmentations, and monitor training metrics in realtime. App automatically converts all labels to rectangles. All training artifacts including model weights will be saved to Team Files and can be easily downloaded. 

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov5/tree/master/supervisely/train" src="https://i.imgur.com/RkVzrLC.png" width="350px"/>

2. [Serve YOLOv5](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fyolov5%252Fsupervisely%252Fserve) - serve model as Rest API service. You can run pretrained model, use custom model weights trained in Supervisely as well as weights trained outside (just upload weights file to Team Files). Thus other apps from Ecosystem can get predictions from the deployed model. Also developers can send inference requiests in a few lines of python code.
   
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov5/tree/master/supervisely/serve" src="https://i.imgur.com/DVONwK8.png" width="350px"/>

3. [Apply NN to images project ](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analise predictions and perform automatic data pre-labeling.   
   
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/tree/master/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" width="350px"/> 

4. [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image. 
   
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/tree/master/annotation-tool" src="https://i.imgur.com/hYEucNt.png" width="350px"/> 

5. [Convert Supervisely to YOLO v5 format](https://ecosystem.supervise.ly/apps/convert-supervisely-to-yolov5-format) - export labeled images project in yolov5 compatible format. 
   
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-supervisely-to-yolov5-format" src="https://i.imgur.com/9cfB1m0.png" width="350px"/> 

6. [Convert YOLO v5 to Supervisely format](https://ecosystem.supervise.ly/apps/convert-yolov5-to-supervisely-format) - import images and yolov5 annotatons to Supervisely.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-yolov5-to-supervisely-format" src="https://i.imgur.com/roiJIE8.png" width="350px"/> 

# For Developers
- you can use sources of [Serve YOLOv5 app](https://github.com/supervisely-ecosystem/yolov5/tree/master/supervisely/serve) as example of how to prepare weights, initialize model and apply it to a folder with images (or to images URLs)
- This apps collection is based on the original YOLOv5 [release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0). Once a next official release is available, all apps will be synchronized with it and also released with the new versions. Before running any app you can choose what version to use. Also Supervisely Team will pull updates from original master branch from time to time.

# Contact & Questions & Suggestions

- for technical support please leave issues, questions or suggestions to original [YOLOv5 repo](https://github.com/ultralytics/yolov5/issues) with the prefix `[Supervisely]`. Our team will try to help.
- also we can chat in slack channel [![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack) 
- if you are interested in Supervisely Enterprise Edition (EE) please send us a [request](https://supervise.ly/enterprise/?demo) or email Yuri Borisov at [sales@supervise.ly](sales@supervise.ly)