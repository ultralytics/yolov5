python train.py \
--cfg "./models/yolov5s_attn.yaml" \
--data "./data/litter_general.yaml" \
--hyp "./data/hyps/hyp_general.yaml" \
--project "./runs/Garbage_yolov5s_attn_raw_$(date +"%d-%m-%Y")" \
--batch-size 16 \
--imgsz 512 \
--epochs 100 \
--device 0