#MODEL_PATH="./runs/CBAM@17_leakyReLU_litter_xinbuju_blend_less50/exp/weights/best.pt"
MODEL_PATH="./runs/Garbage_yolov5s_attn_with_sewage_ar0005_aug_14-02-2022/exp4/weights/"
PT="best.pt"
IMG_SIZE=512
BATCH=1

echo "converting torch model into ONNX"
python ./export.py \
--weights "${MODEL_PATH}${PT}" \
--img-size $IMG_SIZE \
--batch-size $BATCH \
--include 'onnx' \
--opset 10

echo "copy onnx to /onnx/$(date +"%d-%m-%Y")"
cd ./onnx/
mkdir $(date +"%d-%m-%Y")
cd ..
cp "${MODEL_PATH}best.onnx" ./onnx/$(date +"%d-%m-%Y")/aug_best.onnx

