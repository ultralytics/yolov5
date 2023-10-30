# import cv2
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Grayscale, Gaussian blur, Adaptive threshold
# image = cv2.imread(r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\train\exp\lorry15.jpg')
# original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)

# # Perform morph close to merge letters together
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

# # Find contours, contour area filtering, extract ROI
# cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area > 1800 and area < 2500:
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = original[y:y+h, x:x+w]
#         cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

# # Perform text extraction
# ROI = cv2.GaussianBlur(ROI, (3,3), 0)
# data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
# print(data)

# cv2.imshow('ROI', ROI)
# cv2.imshow('close', close)
# cv2.imshow('image', image)
# cv2.waitKey()
# import cv2
# import pytesseract

# Load the image
#image = cv2.imread(r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\train\exp\lorry15.jpg')
#pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # Apply OCR to extract text
# x, y, width, height =   # Replace with your coordinates

# # Crop the region inside the boundary box
# roi = image[y:y + height, x:x + width]

# # Apply OCR to the cropped region
# text = pytesseract.image_to_string(roi)

# # Print or use the extracted text
#print("Text inside the boundary box:", text)
#import torch
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.general import non_max_suppression
# from yolov5.utils.datasets import LoadImages

# # Load a pre-trained YOLOv5 model
# model = attempt_load(r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\train\exp\weights\last.pt', map_location=torch.device('cuda'))  # Load the pre-trained model weights

# # Load and preprocess an image
# img = r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\detect\exp3\lorry_mask.jpg'
# dataset = LoadImages(img)

# # Run inference
# results = model(img)

# # Apply non-maximum suppression to filter results
# results = non_max_suppression(results)

# # Access bounding box coordinates for detected objects
# for det in results[0]:
#     if det is not None:
#         for x1, y1, x2, y2, conf, cls in det:
#             # x1, y1, x2, y2 are the bounding box coordinates
#             print(f"Class {int(cls)} - Bounding Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf}")
# import cv2
# import pytesseract

# image = cv2.imread(r'C:\Users\manik\OneDrive\Pictures\lorry_mask_copy.jpg')
# pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# text = pytesseract.image_to_string(image)

# print("License Plate Text:", text)
import easyocr

#Initialize the OCR reader (English is the default language)
reader = easyocr.Reader(['en'])

#Load the image
image_path = r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\detect\exp9\indian_lorry.jpg'

#Perform OCR on the image
result = reader.readtext(image_path)

#Extract and print the recognized text
# extracted_text = [text[1] for text in result]
# print("\n".join(extracted_text))
# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from PIL import Image
# import torchvision
# # Load a pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Load and preprocess an image
# image = Image.open(r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\detect\exp3\lorry_mask.jpg')
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# image = transform(image)

# # Run inferencezz
# with torch.no_grad():
#     prediction = model([image])

# # Access bounding box coordinates for detected objects
# for box in prediction[0]['boxes']:
#     x_min, y_min, x_max, y_max = box.tolist()
#     print(f"Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
import cv2

# Load your image
image = cv2.imread(r'C:\Users\manik\Downloads\yolov5s\yolov5\runs\detect\exp3\lorry_mask.jpg')

# Define the bounding box coordinates
x_min, y_min, x_max, y_max = 130.548889, 784.011047, 335.168975, 1152.911987

# Draw the bounding box on the image
cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

# Save or display the image with bounding box
cv2.imwrite('output_image.jpg', image)









