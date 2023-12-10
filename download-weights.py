from utils.downloads import attempt_download

p5 = list('nsmlx')  # P5 models
p6 = [f'{x}6' for x in p5]  # P6 models
cls = [f'{x}-cls' for x in p5]  # classification models
seg = [f'{x}-seg' for x in p5]  # classification models

for x in p5 + p6 + cls + seg:
    attempt_download(f'weights/yolov5{x}.pt')