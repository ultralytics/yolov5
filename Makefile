run_interpretability:
	python explainer/explainer.py --weights runs/train/exp8/weights/best.pt \
	 --source /mnt/new_drive/pourmand/KUMC-Harvard/KUMC/train2019/images/ \
	  --method GradCAM --keep-only-topk 1 --crop True --device 0

run_interpretability_old:
	python explainer/explainer.py --weights runs/train/exp8/weights/best.pt \
	 --source /mnt/new_drive/pourmand/KUMC-Harvard/KUMC/train2019/images/ \
	  --method GradCAM --keep-only-topk 1 --crop True --device 0 --use-old-target-method True
	  