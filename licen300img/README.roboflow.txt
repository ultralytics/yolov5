
licenplate detect - v8 2023-05-21 8:50am
==============================

This dataset was exported via roboflow.com on May 21, 2023 at 3:52 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 303 images.
Car-motorcycle are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Resize to 1280x1280 (Fill (with center crop))

The following augmentation was applied to create 3 versions of each source image:
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 5 percent of pixels
