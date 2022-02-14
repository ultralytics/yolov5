EXPERIMENT="Garbage_yolov5s_attn_raw_14-02-2022"

declare -a TESTS=("test" "test_tongzhou" "test_xian" "test_huanqiu" "test_tianan" "test_gangzhuao")

#----------------------
# Evaluation
#----------------------
for i in ${TESTS[@]}; do
    python val.py \
    --data "./data/litter_general.yaml" \
    --test_scene "images/${i}" \
    --weights "./runs/${EXPERIMENT}/exp2/weights/best.pt" \
    --imgsz 512 \
    --conf-thres 0.5 \
    --iou-thres 0.45 \
    --device 0 \
    --save-txt \
    --project "./inference/${EXPERIMENT}" \
    --name ${i} \
    --save-conf
done
