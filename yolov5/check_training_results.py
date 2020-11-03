from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_data_from_path(path):
    with path.open("r") as file:
        data = np.array([list(map(float, line.rstrip().split())) for line in file.readlines()])
    ground_truth, training_data = data[0, 0], data[:, 1:]
    prediction_data = {}
    for i in range(0, training_data.shape[1], 4):
        prediction_data[training_data[0, i]] = training_data[:, i+1:i+4]
    return ground_truth, prediction_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="check_training_results.py", description="Script to evaluate and display the training results on the test data.")
    parser.add_argument('--run-num', type=int, default=0, help='run number to display')
    opt = parser.parse_args()
    root = Path(f"runs/exp{opt.run_num}").resolve()
    data_paths = list(root.glob("training_info_mojo/*.txt"))
    data = []
    ground_truths = []

    for path in data_paths:
        ground_truth, prediction_data = get_data_from_path(path)
        data.append(prediction_data)
        ground_truths.append(ground_truth)
    ground_truths = np.array(ground_truths)

    n_keys = len(data[0])
    final_predictions = {}
    for key in data[0]:
        final_predictions[key] = []
        for d in data:
            final_predictions[key].append(d[key][-1, 0])
    colors = {}
    for key in data[0]:
        colors[key] = np.random.rand(3)
    keys = [0.5, 0.6, 0.7]

    diff = ground_truths - final_predictions[0.5]
    sorted_ind = np.argsort(diff)

    results_str = f"{'GT':<5} {'Diff':<5} {'Data-location'}\n"
    for i in sorted_ind:
        results_str += f"{int(ground_truths[i]):<5} {int(diff[i]):<5} {data_paths[i]}\n"
    with (root / "test_data_results_end_training.txt").open("w") as f:
        f.writelines(results_str)

    n = len(data)
    h = int(np.sqrt(n))
    w = n//h + 1
    fig, axs = plt.subplots(h, w)
    for i in range(n):
        prediction_data = data[i]
        for key in prediction_data:
            if key not in keys:
                continue
            axs[i//w, i % w].plot(ground_truths[i] - prediction_data[key][:, 0], color=colors[key], alpha=0.5, label=key)
        axs[i // w, i % w].set_ylim([-10, 10])

    axs[0, 0].legend()
    axs[-1, w//2].set_xlabel("Epoch")
    axs[h//2, 0].set_ylabel("N_tagged - N_predicted")
    plt.savefig(root / "test_data_training_count_error.pdf")
    plt.clf()

    rand_vector = (np.random.rand(2, len(ground_truths))-0.5)/4
    for key in final_predictions:
        if key not in keys:
            continue
        plt.scatter(ground_truths+rand_vector[0], np.array(final_predictions[key])+rand_vector[1], lw=0, alpha=0.5, label=f"{key}")
    plt.plot([0, np.max(ground_truths)], [0, np.max(ground_truths)])
    plt.legend()
    plt.xlabel("Ground Truth")
    plt.ylabel("N-predicted")
    plt.title("Ground truths vs Predictions on the test data at the end of training")
    plt.savefig(root / "test_data_groundtruth_vs_predictions.pdf")
    plt.clf()
