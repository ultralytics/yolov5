#!/usr/bin/env bash
cd ../
mkdir -p weights

# download official weights
wget https://gh.ddlc.top/https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P weights
# export yolov5s.onnx
python3 export.py --weights weights/yolov5s.pt --include onnx engine  --nms
mv weights/yolov5s.onnx ./examples/yolov5s_nms.onnx
cd examples
trtexec --onnx=./yolov5s_nms.onnx --saveEngine=./yolov5s_nms_fp16.engine --fp16

# result test
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/image1.jpg
python3 trt_infer.py
trtexec --loadEngine=./yolov5s_nms_fp16.engine --verbose --useCudaGraph --noDataTransfers --shapes=images:1x3x640x640
