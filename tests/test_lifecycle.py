from pathlib import Path

from yolov5 import benchmark, detect, export, train, val


# train-> val-> detect -> export -> benchmark
def test_complete_lifecycle():
    # TODO: Create dummy model for test
    dummy_model_cfg = "yolov5s.yaml"

    # train.py run() returns opt
    train_params = train(cfg=dummy_model_cfg, epochs=1)
    best_weight = Path(train_params.save_dir) / 'weights' / 'best.pt'
    assert best_weight.is_file(), "Trained weights not saved"

    # val.py run() returns metrics
    val_score = val(weights=best_weight)
    assert val_score is not None, "Val score is None"

    # detect.py run() returns None
    detect(weights=best_weight)

    exported_weights = export(best_weight, include=("torchscript", ))
    exported_weight = Path(exported_weights[0])
    assert exported_weight.is_file(), "Exported weight was not saved"

    result = benchmark(weights=best_weight)  # pandas dataframe
    print(result.head())
    # TODO: Set assertion for minimum score threshold


if __name__ == "__main__":
    test_complete_lifecycle()
