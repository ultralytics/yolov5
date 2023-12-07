#!/bin/bash
run_file="detect.py"
data_path=/home/aru/yolov5_hdr/datasets/real_safe
save_base_name=safe_10_real_fold
weight_base_path=runs/all/10_real_folds/10_real_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
		echo $save_name
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --conf-thres 0.001 --weights $weight_path --source $data_path --save-txt --save-conf --name $save_name
    done
save_base_name=safe_10_real_10_synth_fold
weight_base_path=runs/all/10_real_10_synth_folds/10_real_10_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --conf-thres 0.001 --weights $weight_path --source $data_path --save-txt --save-conf --name $save_name
    done
save_base_name=safe_10_real_20_synth_fold
weight_base_path=runs/all/10_real_20_synth_folds/10_real_20_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --conf-thres 0.001 --weights $weight_path --source $data_path --save-txt --save-conf --name $save_name
    done
save_base_name=safe_10_real_40_synth_fold
weight_base_path=runs/all/10_real_40_synth_folds/10_real_40_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --conf-thres 0.001 --weights $weight_path --source $data_path --save-txt --save-conf --name $save_name
    done
save_base_name=safe_10_real_80_synth_fold
weight_base_path=runs/all/10_real_80_synth_folds/10_real_80_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --conf-thres 0.001 --weights $weight_path --source $data_path --save-txt --save-conf --name $save_name
    done
