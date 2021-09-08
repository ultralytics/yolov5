<div align="center" markdown>
<img src="https://i.imgur.com/VwuZNID.png"/>

# Export YOLOv5 weights

<p align="center">
  <a href="#Overview">Overview</a>
  <a href="#How-To-Use">How To Use</a>
  <a href="#Infer-models">Infer models</a>
  <a href="Sliding-window-approach">Infer models</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/yolov5/supervisely/export_weights)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/yolov5)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/yolov5/supervisely/export_weights&counter=runs&label=runs&123)](https://supervise.ly)

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
        --image=/path/to/image.{any extension}
        --mode={direct / sliding_window}
        --conf_thresh=0.25
        --iou_thresh=0.45
        --sliding_window_step 1 1 
        --original_model=/path/to/{original_model_name}.pt
        
```
Options to set:

Required for `*.pt`, `*.torchscript.pt`, `*.onnx`
```
  --weights         path to model weights
  --cfgs            path to model cfgs (required for ONNX anf TorchScript models)
  --image           path to model image
  --mode            {direct,sliding_window} inference mode
  --conf_thresh     confidence threshold
  --iou_thresh      intersection over union threshold
  --viz             flag for results visualisation
  --save_path       path to save inference results
```

Additional options for `*.torchscript.pt`, `*.onnx`
```
  --original_model  path to original model to construct meta (required for ONNX anf TorchScript models)
```

if `mode` set to `sliding_window`:
```
  --native               for sliding window approach
  --sliding_window_step  [SLIDING_WINDOW_STEP ...]
```

**TorchScript**
```python
customWeightsPath_torchScript = '/path/to/remote/weights/best.torchscript.pt'
path_to_torch_script_saved_model = download_weights(customWeightsPath_torchScript)
torch_script_model = torch.jit.load(path_to_torch_script_saved_model)
torch_script_model_inference = torch_script_model(tensor)[0]
```
**ONNX**
```python
customWeightsPath_onnx = "/path/to/remote/weights/best.onnx"
path_to_onnx_saved_model = download_weights(customWeightsPath_onnx)
onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
input_name = onnx_model.get_inputs()[0].name
label_name = onnx_model.get_outputs()[0].name
onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]
```

# Sliding window approach

allows to infer hight resolution images:

steps to run:
 - prepare model:
    - download weights. 
    - init model for downloaded weights (use [prepare_model](https://github.com/supervisely-ecosystem/yolov5/blob/2016c53e12c3e22c313e5313143d75eac75f15da/supervisely/export_weights/src/sliding_window.py#L124) function)
 - get and prepare image
    - download image, resize it if it's necessary
    - convert image to model input format(convert torch.Tensor or numpy.ndarray, divide to 255 if values in range 0-255)
 - infer image:
    - use [infer_model](https://github.com/supervisely-ecosystem/yolov5/blob/2016c53e12c3e22c313e5313143d75eac75f15da/supervisely/export_weights/src/sliding_window.py#L156) function
 - visualize results:
    - use [visualize_dets](https://github.com/supervisely-ecosystem/yolov5/blob/2016c53e12c3e22c313e5313143d75eac75f15da/supervisely/export_weights/src/sliding_window.py#L101) function
