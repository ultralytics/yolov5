import os
import sys
import argparse
import json
import yaml
import train, test
import subprocess
from glob import glob
from sklearn.model_selection import train_test_split
from utils import general_utils as g_utils
from utils import file_utils
from utils import coordinates as coord


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main_generate(ini, logger=None):
    g_utils.folder_exists(ini['label_path'], create_=True)

    img_fnames = sorted(g_utils.get_filenames(ini['img_path'], extensions=g_utils.IMG_EXTENSIONS))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    for idx, img_fname in enumerate(img_fnames):
        logger.info(" [GENERATE-YOLO] # Processing {} ({:d}/{:d})".format(img_fname, (idx + 1), len(img_fnames)))

        _, img_core_name, img_ext = g_utils.split_fname(img_fname)
        img = g_utils.imread(img_fname, color_fmt='RGB')
        h, w, c = img.shape

        # Save object info to COCO format
        rst_fpath = os.path.join(ini['label_path'] + img_core_name + '.txt')
        class_no, x_center, y_center, width, height = str(0), str((w/2) / w), str((h/2) / h), str(w/w), str(h/h)
        with open(rst_fpath, 'w') as f:
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
        print(" @ Warning: test text file path, {}, already exists".format(ini["test_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()

    img_fnames = sorted(g_utils.get_filenames(ini['img_path'], extensions=g_utils.IMG_EXTENSIONS))
    img_fnames = [path.replace('data/', '') for path in img_fnames]

    test_ratio = float(ini['test_ratio'])
    train_ratio = 1-test_ratio
    train_img_list, test_img_list = train_test_split(img_fnames,
                                                     test_size=test_ratio, random_state=2000)
    # Save train.txt file
    with open(ini['train_path'], 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open(ini['val_path'], 'w') as f:
        f.write('\n'.join(test_img_list) + '\n')

    logger.info(" [SPLIT] # Train : Test ratio -> {} : {}".format(train_ratio, test_ratio))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(train_img_list), len(test_img_list)))

    # Modify yaml file
    with open(ini['ref_yaml_path'], 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = ini['train_path']
    data['val'] = ini['val_path']

    # Save yaml file
    with open(ini['rst_yaml_path'], 'w') as f:
        yaml.dump(data, f)
        print(data)


    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_train(ini, logger=None):
    logger.info(" [TRAIN] # Train shell : {}".format(ini['train_sh_path']))
    logger.info(" [TRAIN] # Train shell is running ...")

    subprocess.call(ini['train_sh_path'], shell=True, cwd=ini['sh_root_path'])

    return True

def main_test(ini, model_dir=None, logger=None):
    if not model_dir:
        model_dir = max([os.path.join(ini['model_root_path'],d) for d in os.listdir(ini["model_root_path"])],
                        key=os.path.getmtime)
    else:
        model_dir = os.path.join(ini["model_root_path"], model_dir)
    model_path = os.path.join(model_dir, os.path.basename(model_dir) + '-craft' + '.pth')

    test_args = ['--pretrain_model_path', model_path,
                 '--test_img_path', ini['test_img_path'],
                 '--test_gt_path', ini['test_gt_path']]

    return True



def main(args):
    ini = g_utils.get_ini_parameters(os.path.join(_this_folder_, args.ini_fname))
    logger = g_utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE':
        main_generate(ini['GENERATE'], logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini['SPLIT'], logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini['TRAIN'], logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini['TEST'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
        main_test(ini['TEST'], model_dir, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['GENERATE', 'SPLIT', 'TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'TRAIN' # GENERATE / SPLIT / TRAIN / TEST / TRAIN_TEST
INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            # sys.argv.extend(["--model_dir", '200925'])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))