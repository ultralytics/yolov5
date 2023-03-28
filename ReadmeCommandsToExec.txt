<-- Can be wrong. Might need edits. -->

[Training]
python train.py --img 224 --batch 300 --epochs 10 --data fodDetection.yaml --weights yolov5m.pt --hyp hyp.scratch-low.yaml --name FOD

[Detection]
python detect.py --weights runs/train/FOD/weights/best.pt --img 224 --conf 0.25 --source testDB\

[Resume Training]
python train.py --resume {path/to/model_file.pt}

- Ribhu
