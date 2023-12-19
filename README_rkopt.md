# YOLOv5 - RKNN optimize

## Source

  Base on https://github.com/ultralytics/yolov5 (v7.0) with commit id as 915bbf294bb74c859f0b41f1c23bc395014ea679



## What different

With inference result values unchanged, the following optimizations were applied:

- Optimize focus/SPPF block, getting better performance with same result
- Change output node, remove post_process from the model. (post process block in model is unfriendly for quantization)



With inference result got changed, the following optimization was applied:

- Using ReLU as activation layer instead of SiLU（Only valid when training new model）



## How to use

```
# for detection model
python export.py --rknpu --weight yolov5s.pt

# for segmentation model
python export.py --rknpu --weight yolov5s-seg.pt
```

- 'yolov5s.pt'/ 'yolov5s-seg.pt' could be replaced with your model path
- A file name "RK_anchors.txt" would be generated and it would be used for the post_process stage. 
- **NOTICE: Please call with --rknpu, do not changing the default rknpu value in export.py.** 



## Deploy demo

Please refer https://github.com/airockchip/rknn_model_zoo

