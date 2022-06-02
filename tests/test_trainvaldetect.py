from pathlib import Path
from yolov5 import train, val, detect


def test_train_val_detect():
    # TODO: Create dummy model for test
    dummy_model_cfg = "yolov5s.yaml"

    # train.py run() returns opt
    train_params = train(cfg=dummy_model_cfg, epochs=1)
    best_weight = Path(train_params.save_dir) / 'weights' / 'best.pt'
    assert best_weight.is_file(), "Trained weights not found"

    # val.py run() returns metrics
    val_score = val(weights=best_weight)
    assert val_score is not None, "Val score is None"
    
    #detect.py run() returns None
    detect(weights=best_weight)

if __name__ == "__main__":
    test_train_val_detect()
