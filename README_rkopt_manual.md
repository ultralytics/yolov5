# YOLOv5 - RKNN optimize

## Source

  Base on https://github.com/ultralytics/yolov5 with commit id as d3ea0df8b9f923685ce5f2555c303b8eddbf83fd



## What different

Inference result unchanged:

- Optimize focus/SPPF block, getting better performance with same result
- Change output node, remove post_process from the model. (post process is unfriendly in quantization)



Inference result changed:

- Using ReLU as activation layer instead of SiLU（Only valid when training new model）



## How to use

```
python export.py --rknpu {rk_platform} --weight yolov5s.pt
```

- rk_platform support  rk1808, rv1109, rv1126, rk3399pro, rk3566, rk3562, rk3568, rk3588, rv1103, rv1106. (Actually the exported models are the same in spite of the exact platform )

- the 'yolov5s.pt' could be replace with your model path
- A file name "RK_anchors.txt" would be generated and it could be use during doing post_process in the outside. 
- **NOTICE: Please call with --rknpu param, do not changing the default rknpu value in export.py.** 

