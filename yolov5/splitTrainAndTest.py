import argparse
import os
import random

allowed_ext = ['jpg', 'jpeg', 'png']
dir_path = os.path.dirname(os.path.realpath(__file__))


def split_data_set(image_dir, test_size):
    images = []

    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            ext = file.split('.')[-1]
            if ext not in allowed_ext:
                continue
            images.append(os.path.join(subdir, file))

    random.shuffle(images)

    test_size = int(test_size * len(images))
    with open(f"{dir_path}/test.txt", 'w') as f:
        f.write("\n".join(images[:test_size]))
    with open(f"{dir_path}/train.txt", 'w') as f:
        f.write("\n".join(images[test_size:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--test_size', type=float, required=True, help='test size')
    args = parser.parse_args()

    split_data_set(args.dataset, test_size=args.test_size)
