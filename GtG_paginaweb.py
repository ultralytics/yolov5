# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0      # Camera Realsense GTG

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.engine             # TensorRT
Crear base de datos:
    Abrir terminal (misma carpeta donde está el archivo)
        # Verificar flask-sqlachemy (2.5.1)   
        > python3                   
        > from app import db        
        > db.create_all()           
        
Ver si base de datos tiene informacion: 
    Abrir terminal (misma carpeta donde la base de datos)
        # Instalar sqlite3 
        >  sqlite3
        > .open database.db (nombre de la base de datos que queremos leer)
        > .tables               | > .dump
        >  select * from user;

"""

import torch

from flask import *
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from models.common import DetectMultiBackend
from pathlib import Path
from realsense import RealSense #GTG
from utils.dataloaders import LoadStreamsGTG
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
#--------------------------------------------------------------------------------------------------------------------------------
# - Variables de YoloV5
weights='weights/yolov5s-seg.engine'  # model.pt path(s)
device_num = '0' # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz=(640, 640)  # inference size (height, width)
# - Rangos de distancias
result = {'min': '0', 'med': '0', 'max': '0'}
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
#Definción de la base de datos
# Crear la app
app = Flask(__name__)
# Crear la extensión
db = SQLAlchemy(app)
# Crear la encriptación
bcrypt = Bcrypt(app)
# Configuracion de la database SQLite , relativa a la app de la carpeta
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/gtgdatabase.db'
app.config['SECRET_KEY'] = 'mi super llave secreta!'
# Flask_Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
#------------------------------------------------------------------------------------------------
# Clase Usuarios Base de Datos
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(10), nullable=False, unique=True)
    password = db.Column(db.String(20), nullable=False)

# Clase LogIn
class LoginForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=3, max=20)], render_kw={"placeholder": "Usuario"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=3, max=20)], render_kw={"placeholder": "Contraseña"})

    submit = SubmitField('Login')
#-------------------------------------------------------------------------------------------------------------
#Deficiones de las routes para las páginas web
@app.route('/') #Página Web Principal
def index():
   return render_template('video.html')


@app.route('/res',methods = ['POST','GET'])
def res(): #Pagina de regreso tras el envio de los valores de la distancia
	global result
	if request.method == 'POST':
		result = request.form.to_dict()
		return render_template("video.html",result = result)


@app.route('/video_feed')
def video_feed(): #Pagina que regresa al programa del YoloV5
    global result
    params = result
    return Response(generate(params), mimetype = "multipart/x-mixed-replace; boundary=image")


@app.route('/distancias')
@login_required 
def distancias(): #Pagina Configuracion de las distancias
    global t_error, t_min, t_med, t_max
    return render_template('distancias.html', terror = t_error, tmin = t_min, tmed = t_med, tmax= t_max)


@app.route('/login', methods=['GET', 'POST'])
def login():    #Pagina de LogIn
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('distancias'))
    return render_template('login.html', form=form)


@app.route('/logout', methods=['GET', 'POST'])
@login_required 
def logout():   #Pagina de LogOut
    logout_user()
    return redirect(url_for('res'))


@login_manager.user_loader
def load_user(user_id):     #Gestión del usuario
    return User.query.get(int(user_id))
#------------------------------------------------------------------------------------------------------------------
# Definición YoloV5
@smart_inference_mode()
def generate(params):    
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
        #Total time
        dt_total = (dt[0].dt + dt[1].dt +dt[2].dt)
        fps = round((1/(dt_total)),2)
        #-----------------------------------------------------------------------------------------------------------
        # Evaluación distancias pagina web
        global t_error, t_min, t_med, t_max
        dmin = float(params['min'])*1000
        dmed = float(params['med'])*1000
        dmax = float(params['max'])*1000
        if (dmin < 0) or (dmin > dmed) or (dmed > dmax) or (dmin >= dmax):
            #Se cargan valores por defecto
            dmin = 1000; dmed = 2000; dmax = 3000  
            t_error = 'Valores por defecto'
            t_min,t_med, t_max = '1000', '2000', '3000'  
        else:
            t_error = 'Ninguno'
            t_min = dmin ; t_med = dmed ; t_max = dmax
        #------------------------------------------------------------------------------------------------------------       
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3)
            if len(det):
                # Primero escalamos la detección y después aplicamos la máscara
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
                    dist = round((camera.get_distance(cx,cy))*1000) #Expresada en mm
                    #Especificacion del color en funcion de la distancia
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
            # Envio la imagen codificada para el streaming
            if not result:
                continue
            yield(b'--image\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(image) + b'\r\n')

#------------------------------------------------------------------------------------------------------------------
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
