''' Modified utils.datasets.LoadStreams class to ingest a FLIR camera stream, split out to keep in sync with upstream'''
import time
from threading import Thread
import cv2
import numpy as np
import EasyPySpin
from utils.datasets import letterbox

class LoadFLIR:
    def __init__(self, sources=0, img_size=640, stride=32, fps:int=10):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.imgs = [None]
        self.rect = True
        self.sources = [sources] # probably pointless
        cap = EasyPySpin.VideoCapture(int(sources))
        assert cap.isOpened(), f'Failed to open FLIR Camera'
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        _, self.imgs[0] = cap.read()  # guarantee first frame
        thread = Thread(target=self.update, args=([0, cap]), daemon=True)
        print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
        thread.start()
        print('')  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # why is this commented out?
            #_, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            # maaybe we need to tune this better
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
