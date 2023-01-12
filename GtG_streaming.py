# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0      # RealSense

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.engine             # TensorRT
"""
import torch

from flask import *
from models.common import DetectMultiBackend
from realsense import RealSense #GTG#
from utils.dataloaders import LoadStreamsGTG #GTG#
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression,scale_boxes)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
#--------------------------------------------------------------------------------------------------------------------------------
# - Variables de YoloV5
weights='weights/yolov5s-seg.engine'  # model.pt path(s)
device_num = '0' # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz=(640, 640)  # inference size (height, width)
# - Rangos de distancias
dmin = 1000 #Rango minimo color rojo
dmed = 2000 #Rango intermedio color naranja
dmax = 3000 #Rango máximo color verde
# - Medicion de los tiempos de ejecucion de las etapas
dt = (Profile(), Profile(), Profile())
# - Calidad de la imagen
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),95]
# - Gama de colores en RGB
color_negro = (0,0,0) ; color_blanco = (255,255,255)
color_azulceleste = (255,255,0); color_azuloscuro = (255,0,0) ; color_verde = (0 , 255, 0)
color_amarillo = (0,255,255); color_rojo = (0,0,255) ; color_naranja = (26,127,239) 
# - Estilo de letra
font1 = cv2.FONT_HERSHEY_SIMPLEX ; font2 = cv2.FONT_HERSHEY_DUPLEX; font3 = cv2.FONT_HERSHEY_COMPLEX; font4 = cv2.FONT_HERSHEY_TRIPLEX
cv_font = font3
#-----------------------------------------------------------------------------------------------------------------
# - Crear instancia app
app = Flask(__name__)
#------------------------------------------------------------------------------------------------------------
#Definiciones para el streaming
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(run(),
          mimetype = "multipart/x-mixed-replace; boundary=image")
#---------------------------------------------------------------------------------------
#Definicion YoloV5
@smart_inference_mode()
def run():    
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
            pred = non_max_suppression(pred, 0.25, 0.45, 0, False, max_det=100, nm=32)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        #Total time
        dt_total = (dt[0].dt + dt[1].dt +dt[2].dt)
        fps = round((1/(dt_total)),2)
        #------------------------------------------------------------------------------------------------------------       
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3)
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
                    for n in contornos:
                        M = cv2.moments(n)
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00']) ; cy = int(M['m01']/M['m00'])
                    #Distancia del centro a la camara
                    #dist = 2500
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
            result, image = cv2.imencode('.jpg', im0, encode_param) #Codificar la imagen para ser enviada
            if not result:
                continue
            yield(b'--image\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(image) + b'\r\n')
#-------------------------------------------------------------------------------------------
# Programa principal
if __name__ == "__main__":
    # Load model
    device = select_device(device_num)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # DataLoader
    bs = 1  # batch_size
    camera = RealSense()
    dataset = LoadStreamsGTG('0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1, camera = camera)
    
    #Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup  
    
    #Puerto para la ejecución de la aplicación      
    app.run(host ='192.168.0.184', port= 5000, debug=False)
#------------------------------------------------------------------------------------------------- 
