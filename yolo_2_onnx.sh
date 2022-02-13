#MODEL_PATH="./runs/CBAM@17_leakyReLU_litter_xinbuju_blend_less50/exp/weights/best.pt"
MODEL_PATH="./runs/test_openvino/exp4/weights/"
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

echo "copy onnx to /inference/$(date +"%d-%m-%Y")"
mkdir -p ./inference/$(date +"%d-%m-%Y")
cp "${MODEL_PATH}best.onnx" ./inference/$(date +"%d-%m-%Y")

