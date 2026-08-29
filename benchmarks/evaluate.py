# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import json
import pathlib
import sys
import time

import numpy as np

# Ensure root directory is on sys.path for config import
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from src.config import (
        BATCH_SIZE,
        EPOCHS,
        FLIPLR,
        HSV_H,
        HSV_S,
        HSV_V,
        IMG_SIZE,
        LR0,
        LRF,
        MODEL_NAME,
        MOMENTUM,
        MOSAIC,
        OPTIMIZER,
        WARMUP_EPOCHS,
        WEIGHT_DECAY,
    )
except ImportError as e:
    print(f"Error importing configuration from src.config: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing Ultralytics package: {e}")
    sys.exit(1)


def measure_inference_latency(model, img_size: int, num_runs: int = 10) -> float:
    """Measures average inference latency per image in milliseconds."""
    dummy_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Warmup runs
    for _ in range(3):
        model.predict(dummy_img, imgsz=img_size, verbose=False)

    start_time = time.perf_counter()
    for _ in range(num_runs):
        model.predict(dummy_img, imgsz=img_size, verbose=False)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    return avg_latency_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5su mAP@0.5 on COCO128")
    parser.add_argument("--quick", action="store_true", help="Run quick latency screening mode (<=30s)")
    args = parser.parse_args()

    artifacts_dir = pathlib.Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifacts_dir / "results.json"

    print(f"Loading model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    if args.quick:
        print("Running quick mode evaluation (latency screening)...")
        latency_ms = measure_inference_latency(model, img_size=IMG_SIZE, num_runs=5)
        print(f"Measured average inference latency: {latency_ms:.2f} ms")

        results = {
            "schema_version": 1,
            "primary_metric": {"name": "map50", "value": 0.0},
            "secondary_metrics": {
                "latency_ms": round(latency_ms, 2),
            },
            "seed": 42,
        }
    else:
        print(f"Starting training on COCO128 for {EPOCHS} epochs...")
        train_start = time.time()
        model.train(
            data="coco128.yaml",
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            lr0=LR0,
            lrf=LRF,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            optimizer=OPTIMIZER,
            hsv_h=HSV_H,
            hsv_s=HSV_S,
            hsv_v=HSV_V,
            fliplr=FLIPLR,
            mosaic=MOSAIC,
            project="/tmp/yolo_eval",
            name="train_exp",
            exist_ok=True,
            verbose=True,
            seed=42,
        )
        training_duration = time.time() - train_start
        print(f"Training completed in {training_duration:.1f} seconds.")

        print("Evaluating trained model on validation set...")
        val_metrics = model.val(
            data="coco128.yaml",
            imgsz=IMG_SIZE,
            project="/tmp/yolo_eval",
            name="val_exp",
            exist_ok=True,
            verbose=False,
        )

        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        print(f"Validation mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map50_95:.4f}")

        print("Measuring inference latency...")
        latency_ms = measure_inference_latency(model, img_size=IMG_SIZE, num_runs=10)
        print(f"Average inference latency: {latency_ms:.2f} ms")

        results = {
            "schema_version": 1,
            "primary_metric": {"name": "map50", "value": map50},
            "secondary_metrics": {
                "map50_95": round(map50_95, 4),
                "latency_ms": round(latency_ms, 2),
                "training_time_s": round(training_duration, 1),
            },
            "seed": 42,
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
