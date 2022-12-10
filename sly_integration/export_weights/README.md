<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/183672282-11c5ecfd-d760-41b0-be91-a1d629bbd38c.png"/>


# Export YOLOv5 weights

<p align="center">
  <a href="#Overview">Overview</a>
  <a href="#How-To-Use">How To Use</a>
  <a href="#Infer-models">Infer models</a>
  <a href="#Sliding-window-approach">Sliding window approach</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/export_weights)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/yolov5/supervisely/export_weights)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/yolov5/supervisely/export_weights)](https://supervise.ly)

</div>

# Overview

App exports pretrained YOLO v5 model weights to [Torchscript](https://pytorch.org/docs/stable/jit.html?highlight=model%20features)(.torchscript.pt), [ONNX](https://onnx.ai/index.html)(.onnx) formats. 

# How To Run
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/import-mot-format) if it is not there.

**Step 2**: Find your pretrained model weights file in `Team Files`, open file context menue(right click on it) -> `Run App` -> `Export YOLOv5 weights`.

<img src="https://i.imgur.com/uzMlQ2e.png" width="800px"/>

**Step 3**: Press `Run` button. Now application log window will be opened. You can safely close it.

<img src="https://i.imgur.com/zjXgxhg.png"/>

**Step 4**: Converted model files will be placed to source weight file folder:
 - `{source weights filename}.onnx`
 - `{source weights filename}.torchscript.pt`

<img src="https://i.imgur.com/Xk2Gzr0.png"/>

# Infer models

**saved model loading and usage**

Use following command to infer image:

```#!/bin/bash
python ./path/to/inference_demo.py
        --weights=/path/to/weights/name.{pt, torchscript.pt, onnx}
        --cfgs=/path/to/opt.yaml 
        --image=/path/to/image.{any extension: .png, .jpg etc.}
        --mode={direct / sliding_window}
        --conf_thresh=0.25
        --iou_thresh=0.45
        --sliding_window_step 1 1 
        --original_model=/path/to/{original_model_name}.pt
        --save_path=/path/to/image_name.{any extension: .png, .jpg etc.}
```
Options to set:

Required for `*.pt`, `*.torchscript.pt`, `*.onnx`
```
  --weights       |  path to model weights
  --cfgs          |  path to model cfgs (required for ONNX anf TorchScript models)
  --image         |  path to model image
  --mode          |  {direct,sliding_window} inference mode
  --conf_thresh   |  confidence threshold
  --iou_thresh    |  intersection over union threshold
  --viz           |  flag for results visualisation
  --save_path     |  path to save inference results
```

Additional options for `*.torchscript.pt`, `*.onnx`
```
  --original_model  |  path to original model to construct meta (required for ONNX anf TorchScript models)
```

More info about sliding_window approach [here](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/export_weights/README.md#sliding-window-approach)

**Detailed instructions to infer image manually:**

 - TorchScript
```python
path_to_weight = '/path/to/weights/best.torchscript.pt'
torch_script_model = torch.jit.load(path_to_weight)
torch_script_model_inference = torch_script_model(tensor)[0]
```
 - ONNX
```python
path_to_weight = "/path/to/weights/best.onnx"
onnx_model = rt.InferenceSession(path_to_weight)
input_name = onnx_model.get_inputs()[0].name
label_name = onnx_model.get_outputs()[0].name
onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]
```

# Sliding window approach
More info about Sliding window approach [here](https://github.com/supervisely-ecosystem/yolov5/blob/master/supervisely/export_weights/src/inference_demo.py#L66)

if `mode` set to `sliding_window`:
```
  --native               sliding window approach marker
  --sliding_window_step  [SLIDING_WINDOW_STEP ...]
```
 - for `native` cases of sliding window approach: 
    - if set to `True` - NMS applied immediately to each window inference result.
    - if set to `False` - NMS applied to the whole sliding window result set.
