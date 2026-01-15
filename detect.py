# ============================================
# detect.py - Animal Detection System
# تحذير فوري عند أول كشف
# ============================================

import random
import sys
import threading
import time
from pathlib import Path

import cv2
import pygame
import torch
from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# ---------------- Paths ----------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------------- Firebase ----------------
try:
    from firebase_alerts import broadcast_animal_alert

    print("✅ Firebase alerts connected")
except ImportError:
    print("❌ firebase_alerts.py not found")
    sys.exit(1)

# ---------------- Sound ----------------
pygame.mixer.init()


def play_alert_sound():
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"⚠️ Sound error: {e}")


# ---------------- Alert Control ----------------
last_alert_time = 0
GLOBAL_COOLDOWN = 5  # ثواني (غيّرها لـ 0 لو بدك بكل فريم)

TARGET_ANIMALS = ["camel", "cow", "dog", "horse", "sheep"]


# ---------------- Main ----------------
@smart_inference_mode()
def run(weights=ROOT / "best.pt", source="0", imgsz=(640, 640), conf_thres=0.6, iou_thres=0.45, device=""):
    global last_alert_time

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    print("📛 Model classes:")
    for k, v in names.items():
        print(k, v)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=model.pt)
    print(f"🚀 Running on source: {source}")

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        preds = non_max_suppression(model(im), conf_thres, iou_thres)

        for i, det in enumerate(preds):
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3)

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    label = names[int(cls)].lower()

                    if label not in TARGET_ANIMALS:
                        continue

                    # عتبة ثقة مباشرة
                    if label == "dog" and conf < 0.85:
                        continue
                    if label != "dog" and conf < 0.80:
                        continue

                    annotator.box_label(xyxy, f"{label} {conf:.2f}")

                    # --------- ALERT IMMEDIATELY ---------
                    now = time.time()
                    if now - last_alert_time >= GLOBAL_COOLDOWN:
                        last_alert_time = now
                        dist = random.randint(100, 500)

                        print(f"⚠️ ALERT: {label} detected ({dist}m)")

                        threading.Thread(target=play_alert_sound, daemon=True).start()

                        threading.Thread(target=broadcast_animal_alert, args=(label, dist), daemon=True).start()

            cv2.imshow("Animal Detection - Press Q to quit", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return


# ---------------- Entry ----------------
if __name__ == "__main__":
    run(source="1")  # غيّرها 0 أو 1 حسب الكاميرا
