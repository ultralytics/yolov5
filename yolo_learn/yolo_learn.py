import os
import sys
import argparse
import json
import yaml
import train, test
import subprocess
from pprint import pprint
from glob import glob
from sklearn.model_selection import train_test_split
from utils import general_utils as g_utils
# from utils import file_utils
# from utils import coordinates as coord


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]
_project_folder_ = os.path.abspath(os.path.join(_this_folder_, os.pardir))


def main_crop(ini, logger=None):
    g_utils.folder_exists(ini['img_path'], create_=True)

    raw_path = os.path.join(_project_folder_, ini['raw_path'])
    ann_path = os.path.join(_project_folder_, ini['ann_path'])
    raw_fnames = sorted(g_utils.get_filenames(raw_path, extensions=g_utils.IMG_EXTENSIONS))
    ann_fnames = sorted(g_utils.get_filenames(ann_path, extensions=g_utils.META_EXTENSION))
    logger.info(" [CROP] # Total file number to be processed: {:d}.".format(len(raw_fnames)))

    for idx, raw_fname in enumerate(raw_fnames):
        logger.info(" [CROP] # Processing {} ({:d}/{:d})".format(raw_fname, (idx + 1), len(raw_fnames)))

        _, raw_core_name, raw_ext = g_utils.split_fname(raw_fname)
        img = g_utils.imread(raw_fname, color_fmt='RGB')

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = g_utils.split_fname(ann_fname)
        if ann_core_name == raw_core_name + raw_ext:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        # Extract crop position
        object_cnt = 0
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != ini['object_class']:
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            try:
                crop_img = img[y_min:y_max, x_min:x_max]
            except TypeError:
                logger.error(" [CROP] # Crop error : {}".format(raw_fname))
                logger.error(" [CROP] # Error pos : {}, {}, {}, {}".format(x_min, x_max, y_min, y_max))
                pass

            # Save cropped image
            rst_fpath = os.path.join(_project_folder_, ini['img_path'] + raw_core_name + '_' + str(object_cnt) + raw_ext)
            g_utils.imwrite(crop_img, rst_fpath)
            object_cnt += 1

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_generate(ini, logger=None):
    label_path = os.path.join(_project_folder_, ini['label_path'])
    g_utils.folder_exists(label_path, create_=True)

    raw_path = os.path.join(_project_folder_, ini['raw_path'])
    ann_path = os.path.join(_project_folder_, ini['ann_path'])
    raw_fnames = sorted(g_utils.get_filenames(raw_path, extensions=g_utils.IMG_EXTENSIONS))
    ann_fnames = sorted(g_utils.get_filenames(ann_path, extensions=g_utils.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(raw_fnames)))

    for idx, raw_fname in enumerate(raw_fnames):
        logger.info(" [GENERATE] # Processing {} ({:d}/{:d})".format(raw_fname, (idx + 1), len(raw_fnames)))

        _, raw_core_name, raw_ext = g_utils.split_fname(raw_fname)
        img = g_utils.imread(raw_fname, color_fmt='RGB')
        h, w, c = img.shape

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = g_utils.split_fname(ann_fname)
        if ann_core_name == raw_core_name + raw_ext:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        # Extract crop position
        object_names = ini['object_names'].replace(' ', '').split(',')
        object_type = ini['object_type']
        for obj in objects:
            object_name = obj['classTitle']

            if object_name not in object_names:
                continue

            if object_type == 'problem':
                if object_name == 'problem_intro':
                    class_num = 0
                elif object_name == 'problem_whole':
                    class_num = 1
                elif object_name == 'problem_text':
                    class_num = 2
            elif object_type == 'graph':
                class_num = 0

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            # Save object info to COCO format
            rst_fpath = os.path.join(_project_folder_, ini['label_path'] + raw_core_name + '.txt')
            class_no, x_center, y_center, width, height = \
                str(class_num), str(((x_max+x_min)/2) / w), str(((y_max+y_min)/2) / h), str((x_max-x_min)/w), str((y_max-y_min)/h)
            with open(rst_fpath, 'a') as f:
                strResult = "{} {} {} {} {}\r\n".format(class_no, x_center, y_center, width, height)
                f.write(strResult)
            pass

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_split(ini, logger=None):
    g_utils.folder_exists(ini['img_path'], create_=False)

    if g_utils.file_exists(ini['train_path']):
        print(" @ Warning: train text file path, {}, already exists".format(ini["train_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
    if g_utils.file_exists(ini['val_path']):
        print(" @ Warning: test text file path, {}, already exists".format(ini["val_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()


    # Apply symbolic link for img path
    raw_path = os.path.join(_project_folder_, ini['raw_path'])
    img_path = os.path.join(_project_folder_, ini['img_path'])
    g_utils.folder_exists(img_path, create_=True)

    img_fnames = sorted(g_utils.get_filenames(img_path, extensions=g_utils.IMG_EXTENSIONS))
    if len(img_fnames) == 0:
        sym_cmd = "ln -s {} {}".format(raw_path + '*', img_path) # to all files
        subprocess.call(sym_cmd, shell=True)

    img_fnames = sorted(g_utils.get_filenames(img_path, extensions=g_utils.IMG_EXTENSIONS))

    test_ratio = float(ini['test_ratio'])
    train_ratio = 1-test_ratio
    train_img_list, test_img_list = train_test_split(img_fnames,
                                                     test_size=test_ratio, random_state=2000)
    # Save train.txt file
    train_path = os.path.join(_project_folder_, ini['train_path'])
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    val_path = os.path.join(_project_folder_, ini['val_path'])
    with open(val_path, 'w') as f:
        f.write('\n'.join(test_img_list) + '\n')

    logger.info(" [SPLIT] # Train : Test ratio -> {} : {}".format(train_ratio, test_ratio))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(train_img_list), len(test_img_list)))

    # Modify yaml file
    ref_yaml_path = os.path.join(_project_folder_, ini['ref_yaml_path'])
    with open(ref_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.join(_project_folder_, ini['train_path'])
    data['val'] = os.path.join(_project_folder_, ini['val_path'])
    data['names'] = ini['object_names'].replace(' ', '').split(',')
    data['nc'] = len(data['names'])

    # Save yaml file
    rst_yaml_path = os.path.join(_project_folder_, ini['rst_yaml_path'])
    with open(rst_yaml_path, 'w') as f:
        yaml.dump(data, f)
        pprint(data)

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_train_test(ini, logger=None):
    img_size, batch, epoch, data_yaml_path, model_yaml_path, model_weight_path, rst_dir_name =\
        ini['img_size'], ini['batch'], ini['epoch'], ini['data_yaml_path'], ini['model_yaml_path'], ini['model_weight_path'], ini['rst_dir_name']

    train_cmd = ini['train_py_cmd']
    train_args = ['--img', img_size, '--batch', batch,
                  '--epoch', epoch, '--data', data_yaml_path,
                  '--cfg', model_yaml_path, '--weight',  model_weight_path,
                  '--name', rst_dir_name]

    for arg in train_args:
        train_cmd += ''.join([' ', arg])

    logger.info(" [TRAIN] # Train shell cmd : {}".format(train_cmd))
    subprocess.call(train_cmd, shell=True, cwd=ini['train_root_path'])      # train by python
    # subprocess.call(train_cmd, shell=True, cwd=ini['train_root_path'])    # train by shell

    return True


def main(args):
    ini = g_utils.get_ini_parameters(os.path.join(_this_folder_, args.ini_fname))
    logger = g_utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'CROP':
        main_crop(ini['CROP'], logger=logger)
    elif args.op_mode == 'GENERATE':
        main_generate(ini['GENERATE'], logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini['SPLIT'], logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        main_train_test(ini['TRAIN_TEST'], logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_mode", required=True, choices=['PROBLEM', 'GRAPH'], help="Target mode")
    parser.add_argument("--op_mode", required=True, choices=['CROP', 'GENERATE', 'SPLIT', 'TRAIN_TEST'], help="Operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
TARGET = 'GRAPH' # PROBLEM / GRAPH
OP_MODE = 'TRAIN_TEST' # GENERATE / SPLIT / TRAIN_TEST (CROP은 별도의 기능)
if TARGET == 'PROBLEM':
    INI_FNAME = _this_basename_ + "_problem.ini"
elif TARGET == 'GRAPH':
    INI_FNAME = _this_basename_ + "_graph.ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--target_mode", TARGET])
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            # sys.argv.extend(["--model_dir", '200925'])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))