#!/bin/bash
task_name="test"
run_file="test.py"
save_base_name=80_real_fold
yaml_base_path=/home/aru/yolov5_hdr/yaml/real_test/real_test_fold
weight_base_path=runs/all/80_real_folds/real_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
		echo $save_name
		yaml_path=$yaml_base_path$i.yaml
		echo $yaml_path
                imgname=/home/aru/run5/40/renderImg/img_${obj_name}_${base_name}
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --weights $weight_path --data $yaml_path --batch 16 --task $task_name --save-txt --save-json --save-conf --name $save_name
    done
save_base_name=r4_5_80_real_10_synth_fold
yaml_base_path=/home/aru/yolov5_hdr/yaml/real_test/real_test_fold
weight_base_path=runs/all/r4_5_80_real_10_synth_folds/r4_5_80_real_10_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                yaml_path=$yaml_base_path$i.yaml
                echo $yaml_path
                imgname=/home/aru/run5/40/renderImg/img_${obj_name}_${base_name}
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --weights $weight_path --data $yaml_path --batch 16 --task $task_name --save-txt --save-json --save-conf --name $save_name
    done
save_base_name=r4_5_80_real_20_synth_fold
yaml_base_path=/home/aru/yolov5_hdr/yaml/real_test/real_test_fold
weight_base_path=runs/all/r4_5_80_real_20_synth_folds/r4_5_80_real_20_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                yaml_path=$yaml_base_path$i.yaml
                echo $yaml_path
                imgname=/home/aru/run5/40/renderImg/img_${obj_name}_${base_name}
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --weights $weight_path --data $yaml_path --batch 16 --task $task_name --save-txt --save-json --save-conf --name $save_name
    done
save_base_name=r4_5_80_real_40_synth_fold
yaml_base_path=/home/aru/yolov5_hdr/yaml/real_test/real_test_fold
weight_base_path=runs/all/r4_5_80_real_40_synth_folds/r4_5_80_real_40_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                yaml_path=$yaml_base_path$i.yaml
                echo $yaml_path
                imgname=/home/aru/run5/40/renderImg/img_${obj_name}_${base_name}
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --weights $weight_path --data $yaml_path --batch 16 --task $task_name --save-txt --save-json --save-conf --name $save_name
    done
save_base_name=r4_5_80_real_80_synth_fold
yaml_base_path=/home/aru/yolov5_hdr/yaml/real_test/real_test_fold
weight_base_path=runs/all/r4_5_80_real_80_synth_folds/r4_5_80_real_80_synth_fold
for i in {1..5}
    do
                save_name=$save_base_name$i
                echo $save_name
                yaml_path=$yaml_base_path$i.yaml
                echo $yaml_path
                imgname=/home/aru/run5/40/renderImg/img_${obj_name}_${base_name}
                weight_path=$weight_base_path$i/weights/best.pt
                echo $weight_path
                python $run_file --img 1024 --weights $weight_path --data $yaml_path --batch 16 --task $task_name --save-txt --save-json --save-conf --name $save_name
    done
