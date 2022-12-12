import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import train
from train import ROOT


def test_trials():

    weights_path = ROOT / 'yolov5n6.pt'
    data_path = ROOT / 'data/coco128.yaml'
    hyp_path = ROOT / 'tests/hyp.test-hyps.yaml'
    project_path = ROOT / 'runs/tests'
    task_name = 'tests_optuna_first_trial'

    test_opt = dict(
        weights=weights_path,
        data=data_path,
        hyp=hyp_path,
        project=project_path,
        name=task_name,
        nosave=True,
        noplots=True,
        exist_ok=True,
        autoanchor=False,
        # small batch size is used for training instability
        batch_size=32,
        epochs=3,
        evolve=200,
        no_augs_evolving=True,
        optuna=True,
        # set default clearml name
        clearml_project='optuna_tests',
        clearml_task='test_trials',
    )

    opt = train.run(**test_opt)
    hyp_evolve_path = Path(opt.save_dir) / 'hyp_evolve.yaml'
    with open(hyp_evolve_path, errors='ignore') as f:
        results = f.readlines()

    for result_line in results:
        if 'Best generation' in result_line:
            assert ' 0' not in result_line, 'Best generation is 0'

    return opt.save_dir


def test_optuna_gridsearch():
    weights_path = ROOT / 'yolov5n6.pt'
    data_path = ROOT / 'data/coco128.yaml'
    hyp_path = ROOT / 'tests/hyp.test-hyps.yaml'
    project_path = ROOT / 'runs'
    task_name = 'tests/tests_optuna_grid_search'

    test_opt = dict(
        weights=weights_path,
        data=data_path,
        hyp=hyp_path,
        project=project_path,
        name=task_name,
        nosave=True,
        noval=True,
        noplots=True,
        exist_ok=True,
        noautoanchor=True,
        # small batch size is used for training instability
        batch_size=64,
        imgsz=640,
        epochs=1,
        optuna=False,
        # set default clearml name
        clearml_project='optuna_tests',
        clearml_task='test_gridsearch',
    )

    hyp_path = ROOT / 'tests/hyp.test-hyps.yaml'
    with open(hyp_path, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    grid_cell_num = 4
    lr0_grid = np.linspace(1e-7, 0.07, grid_cell_num)
    lrf_grid = np.linspace(0.01, 1.0, grid_cell_num)
    weight_decay_grid = np.linspace(0.0, 0.0001, grid_cell_num)
    warmup_epochs_grid = np.linspace(0.5, 5.0, grid_cell_num)

    for grid_hyps in product(lr0_grid, lrf_grid, weight_decay_grid, warmup_epochs_grid):
        hyp['lr0'] = float(grid_hyps[0])
        hyp['lrf'] = float(grid_hyps[1])
        hyp['weight_decay'] = float(grid_hyps[2])
        hyp['warmup_epochs'] = float(grid_hyps[3])

        test_opt['hyp'] = hyp
        opt = train.run(**test_opt)

    grid_search_result = pd.read_csv(Path(opt.save_dir) / 'results.csv')

    optuna_result = pd.read_csv(project_path / 'tests/tests_optuna_first_trial' / 'results.csv')
    map_str = 'metrics/mAP_0.5:0.95'
    assert max(grid_search_result[map_str]) < max(optuna_result[map_str])


if __name__ == '__main__':
    test_trials()
    test_optuna_gridsearch()
