# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0      # webcam
    $ python segment/predict.py --weights yolov5s-seg.pt --source 4      # RealSense Camera port USB
    $ python segment/predict.py --weights yolov5s-seg.pt --source 5      # GTG program

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.engine             # TensorRT
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import tkinter as tk
import tkinter.font as font
from tkinter import *
from PIL import Image, ImageTk
import threading
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from realsense import RealSense #GTG
from utils.dataloaders import LoadStreams, LoadStreamsGTG
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
#--------------------------------------------------------------------------------------------------------------------------------
#Implementacion cuestiones graficas
# - Gama de colores en RGB
color_negro = (0,0,0) ; color_blanco = (255,255,255)
color_azulceleste = (255,255,0); color_azuloscuro = (255,0,0) ; color_verde = (0 , 255, 0)
color_amarillo = (0,255,255); color_rojo = (0,0,255) ; color_naranja = (26,127,239) 
# - Estilo de letra
font1 = cv2.FONT_HERSHEY_SIMPLEX ; font2 = cv2.FONT_HERSHEY_DUPLEX; font3 = cv2.FONT_HERSHEY_COMPLEX; font4 = cv2.FONT_HERSHEY_TRIPLEX
cv_font = font3
#**Inicializacion de variables**
#-Rangos de distancias
dmin = 1000 #Rango minimo color rojo
dmed = 2000 #Rango intermedio color naranja
dmax = 3000 #Rango máximo color verde
#---------------------------------------------------------------------------------
@smart_inference_mode()
def run(
    weights='weights/yolov5s-seg.engine',  # model.pt path(s)
    source=5,  # file/dir/URL/glob/screen/0(webcam)
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=100,  # maximum detections per image
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=0,  # filter by class: --class 0, or --class 0 2 3
):
    #Selection of the source
    source = str(source)
    webcam = source.isnumeric()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        if int(source) >= 5:
            camera = RealSense()
            dataset = LoadStreamsGTG(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1, camera = camera)
        else:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)   

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        #Conversion of the image
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() # uint8 to fp16
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
            pred, proto = model(im, augment=True, visualize=False)[:2]
        # NMS
        with dt[2]:
            #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det, nm=32)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        #Total time
        dt_total = (dt[0].dt + dt[1].dt +dt[2].dt)
        fps = round((1/(dt_total)),2)
        #------------------------------------------------------------------------------------------------------------       
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # scale bbox first the crop masks
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                mascaras = torch.as_tensor(masks*255, dtype=torch.uint8).cpu().numpy() #De esta manera se pueden visualizar las mascaras para tratar los contonos                
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    #Calculamos el centroide de la máscara mediante la deteccion de contornos
                    contornos, jerarquias = cv2.findContours(mascaras[j], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cx, cy = 0,0
                    #Se hace un bucle de los contornos para evitar oscilaciones en el punto medio
                    for n in contornos:
                        M = cv2.moments(n)
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00']) ; cy = int(M['m01']/M['m00'])
                    #print("CX: ", cx) ; print("CY: ", cy) #Verificar valor de las coordenadas del centroide
                    #Distancia del centro a la camara
                    dist = round((camera.get_distance(cx,cy))*1000) #Expresada en mm
                    #Especificacion del color en funcion de la distancia
                    '''
                        color_select: Seleccion del color del bounding box en RGB
                        color_mask: Seleccion del color de la máscara con un número que se modificó internamente
                    '''
                    if dist >= 0 and dist < dmin:
                        color_select, color_mask = color_rojo, 0
                    elif dist >= dmin and dist < dmed:
                        color_select, color_mask = color_amarillo, 1
                    else:
                        color_select, color_mask = color_verde, 2
                    #Dibujamos el centro calculado
                    cv2.circle(im0, (cx,cy), radius=5, color=color_select, thickness=-1)
                    #Dibujamos el recangulo del detector de la persona
                    label = f'd = ' +str(dist) + 'mm'
                    annotator.box_label(xyxy, label, color= color_select,txt_color=color_negro)
                    #Mask plotting
                    annotator.masks(masks, colors=[colors(color_mask, True) for x in det[:, 5]],
                                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() / 255)
            #------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Visualizacion de los resultados
            cv2.putText(im0, text="FPS: "+ str(fps), org=(15,30),fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color=(0,0,0), thickness=1, lineType = cv2.LINE_AA)
            cv2.imshow(str(source), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        # Print time
        # LOGGER.info(f"{'Conversion: '}{dt[0].dt * 1E3:.1f}ms")
        # LOGGER.info(f"{'Inferencia: '}{dt[1].dt * 1E3:.1f}ms")
        # LOGGER.info(f"{'NMS: '}{dt[2].dt * 1E3:.1f}ms")
        # LOGGER.info(f"{'Tiempo Total: '}{dt_total * 1E3:.1f}ms - {1/(dt_total):.1f}fps")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-seg.engine', help='model path(s)')
    parser.add_argument('--source', type=str, default=4, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


#Creacion de hilos de la pantalla principal
def thread_start():
    thread_1.start()
    
    
def thread_finish():
    root.destroy()
#-------------------------------------------------------------------------------------------------
#Realización de una ventana
root = tk.Tk()
t = tk.StringVar()
thread_1 = threading.Thread(target=run)
thread_1.daemon = True # die when the main thread dies
#Configuracion pantalla de inicio
root.title('Inicio') ; root.geometry("400x275")
myCanvas = Canvas(root) ; myCanvas.pack()
bg_full= Image.open("GTG.png")
bg_resize= bg_full.resize((180,280))
bg = ImageTk.PhotoImage(bg_resize)
label1 = Label( root, image = bg) 
label1.place(x = 0, y = 0) 
my_font = font.Font(root, family="Hello", size = 28, weight="bold")
textos_font = font.Font(root, size = 10, weight="bold")
textos_label = font.Font(root, size = 8)
#-Configuración botones
Start_button = tk.Button(text='START', bg = 'green', font=my_font, command = thread_start)
Start_button.place(x = 190, y = 20, height = 40, width = 200)
Finish_button = tk.Button(text='FINISH', bg='red', font=my_font, command = thread_finish)
Finish_button.place(x = 190, y = 80, height = 40, width = 200)
#-------------------------------------------------------------------------------------------

root.mainloop()
