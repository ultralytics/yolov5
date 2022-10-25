source /repos/LMI_AI_Solutions/lmi_ai.env

python3 -m yolov5.export --weights /app/trained-inference-models/2022-08-18_1024_512_300_s.pt --imgsz 512 1024 --include engine --half --device 0
python3 /app/infer_trt.py --engine /app/trained-inference-models/2022-08-18_1024_512_300_s.engine --imsz 512 1024 --path_imgs /app/images --path_out /app/outputs
