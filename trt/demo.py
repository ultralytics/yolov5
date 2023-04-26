import cv2
import sys
import argparse

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='./weights/yolov5s-simple.trt', help='trt engine file path', required=False)
    parser.add_argument('-i', '--image', default='./data/images/bus.jpg', help='image file path', required=False)
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args.model, letter_box=True)
    visualizer = Visualizer()

    img = cv2.imread(args.image)

    # inference
    output = processor.detect(img)

    # final results
    pred = processor.post_process(output, img.shape, conf_thres=0.5)

    print('Detection result: ')
    for item in pred.tolist():
        print(item)

    visualizer.draw_results(img, pred[:, :4], pred[:, 4], pred[:, 5])



if __name__ == '__main__':
    main()   
