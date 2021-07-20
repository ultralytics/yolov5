"""
Utility functions for working with the YOLOv3 models.

################
Command Help:
usage: utility.py [-h] {strip} ...

Utility functions for working with the YOLOv3 models

positional arguments:
  {strip}

optional arguments:
  -h, --help  show this help message and exit

################
Strip Command Help:
usage: utility.py strip [-h] weights

Strip the extra information from a models checkpoint for training from scratch

positional arguments:
  weights     weights path

optional arguments:
  -h, --help  show this help message and exit
"""

import argparse

from utils.general import strip_optimizer

STRIP_COMMAND = "strip"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility functions for working with the YOLOv3 models")
    subparsers = parser.add_subparsers(dest="command")
    strip_subparser = subparsers.add_parser(
        STRIP_COMMAND,
        description="Strip the extra information from a models checkpoint for training from scratch",
    )
    strip_subparser.add_argument('weights', type=str, help='weights path')
    args = parser.parse_args()

    if args.command == STRIP_COMMAND:
        print(f"stripping extras from {args.weights}")
        strip_optimizer(args.weights)
    else:
        raise ValueError(f"unknown command given of {args.command}")
