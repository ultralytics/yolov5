import base64
import io

from IPython.display import display, Javascript
from google.colab.output import eval_js
import numpy as np
from PIL import Image

from models.experimental import *
from utils.datasets import *
from utils.utils import *

class ColabWebcam:
    
    def __init__(self, weights = 'yolov5s.pt', img_size = 640, conf_thres = 0.4, 
                  iou_thres = 0.5, classes = None, agnostic_nms = True):

        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms

        # Initialize
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
        self.half = self.device != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16
            
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def init_js_video(self):
        js = Javascript('''
                  var video;
                  var div = null;
                  var stream;
                  var captureCanvas;
                  var imgElement;
                  var labelElement;
                  
                  var pendingResolve = null;
                  var shutdown = false;
            
                  function removeDom() {
                      stream.getVideoTracks()[0].stop();
                      video.remove();
                      div.remove();
                      video = null;
                      div = null;
                      stream = null;
                      imgElement = null;
                      captureCanvas = null;
                      labelElement = null;
                  }
            
                  function onAnimationFrame() {
                      if (!shutdown) {
                          window.requestAnimationFrame(onAnimationFrame);
                      }
                      if (pendingResolve) {
                          var result = "";
                          if (!shutdown) {
                              captureCanvas.getContext('2d').drawImage(video, 0, 0, ''' + str(self.img_size) + ''', ''' + str(self.img_size) + ''');
                              result = captureCanvas.toDataURL('image/jpeg', 0.8)
                          }
                          var lp = pendingResolve;
                          pendingResolve = null;
                          lp(result);
                      }
                  }
            
                  async function createDom() {
                      if (div !== null) {
                          return stream;
                      }

                      div = document.createElement('div');
                      div.style.border = '2px solid black';
                      div.style.padding = '10px';
                      div.style.width = '100%';
                      div.style.maxWidth = '700px';
                      div.style.display = 'inline-block';
                      document.body.appendChild(div);
                  
                      const modelOut = document.createElement('div');
                      modelOut.innerHTML = "<span>Status:</span>";
                      labelElement = document.createElement('span');
                      labelElement.innerText = 'No data';
                      labelElement.style.fontWeight = 'bold';
                      modelOut.appendChild(labelElement);
                      div.appendChild(modelOut);
                      
                      video = document.createElement('video');
                      video.style.display = 'block';
                      video.style.float = 'left'
                      video.style.marginRight = '10px'
                      video.style.marginBottom = '10px'
                      video.width = div.clientWidth - 100;
                      video.setAttribute('playsinline', '');
                      video.onclick = () => { shutdown = true; };
                      stream = await navigator.mediaDevices.getUserMedia({video: { facingMode: "environment"}});
                      div.appendChild(video);

                      imgElement = document.createElement('img');
                      imgElement.style.position = 'absolute';
                      imgElement.style.zIndex = 1;
                      div.appendChild(imgElement);
                    
                      const instruction = document.createElement('button');
                      instruction.innerHTML = 'Stop Video'
                      instruction.style.fontSize = 'medium'
                      div.appendChild(instruction);
                      instruction.onclick = () => { shutdown = true; };
                    
                      video.srcObject = stream;
                      await video.play();

                      captureCanvas = document.createElement('canvas');
                      captureCanvas.width = ''' + str(self.img_size) + '''; //video.videoWidth;
                      captureCanvas.height = ''' + str(self.img_size) + '''; //video.videoHeight;
                      window.requestAnimationFrame(onAnimationFrame);
                    
                      return stream;
                  }

                  async function takePhoto(label, imgData) {
                      if (shutdown) {
                        removeDom();
                        shutdown = false;
                        return '';
                      }

                      var preCreate = Date.now();
                      stream = await createDom();
                      
                      var preShow = Date.now();
                      if (label != "") {
                        labelElement.innerHTML = label;
                      }
                    
                      if (imgData != "") {
                        var videoRect = video.getClientRects()[0];
                        imgElement.style.top = videoRect.top + "px";
                        imgElement.style.left = videoRect.left + "px";
                        imgElement.style.width = videoRect.width + "px";
                        imgElement.style.height = videoRect.height + "px";
                        imgElement.src = imgData;
                      }
              
                      var preCapture = Date.now();
                      var result = await new Promise(function(resolve, reject) {
                        pendingResolve = resolve;
                      });
                      shutdown = false;
              
                      return {'create': preShow - preCreate, 
                              'show': preCapture - preShow, 
                              'capture': Date.now() - preCapture,
                              'img': result};
                  }
        ''')

        display(js)

    def js_reply_to_image(self, js_reply):
        """
        input: 
              js_reply: JavaScript object, contain image from webcam

        output: 
              image_array: image array RGB size img_size x img_size from webcam
        """
        jpeg_bytes = base64.b64decode(js_reply['img'].split(',')[1])
        image_PIL = Image.open(io.BytesIO(jpeg_bytes))
        image_array = np.array(image_PIL)

        return image_array

    def get_drawing_array(self, image_array): 
        """
        input: 
              image_array: image array RGB size img_size x img_size from webcam

        output: 
              drawing_array: image RGBA size img_size x img_size only contain bounding box and text, 
                                  channel A value = 255 if the pixel contains drawing properties (lines, text) 
                                  else channel A value = 0
        """
        drawing_array = np.zeros([self.img_size,self.img_size,4], dtype=np.uint8)

        # Padded resize
        img = letterbox(image_array, new_shape=self.img_size)[0]

        # Convert
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()   # uint8 to fp16/32
        img /= 255.0  # (0 - 255) to (0.0 - 1.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Process detections
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_array.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, drawing_array, label=label, color=self.colors[int(cls)])

        drawing_array[:,:,3] = (drawing_array.max(axis = 2) > 0 ).astype(int) * 255

        return drawing_array

    def drawing_array_to_bytes(self, drawing_array):
        """
        input: 
              drawing_array: image RGBA size img_size x img_size 
                                  contain bounding box and text from yolo prediction, 
                                  channel A value = 255 if the pixel contains drawing properties (lines, text) 
                                  else channel A value = 0

        output: 
              drawing_bytes: string, encoded from drawing_array
        """
        drawing_PIL = Image.fromarray(drawing_array, 'RGBA')
        iobuf = io.BytesIO()
        drawing_PIL.save(iobuf, format='png')
        drawing_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(iobuf.getvalue()), 'utf-8')))
        return drawing_bytes

    def start_webcam(self):
        self.init_js_video()
        label_html = 'Capturing...'
        img_data = ''
        while True:
            js_reply = eval_js('takePhoto("{}", "{}")'.format(label_html, img_data))

            if not js_reply:
                break

            image = self.js_reply_to_image(js_reply)
            drawing_array = self.get_drawing_array(image) 
            drawing_bytes = self.drawing_array_to_bytes(drawing_array)
            img_data = drawing_bytes