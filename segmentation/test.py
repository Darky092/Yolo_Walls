from ultralytics import YOLO
import cv2

model = YOLO('runs/segment/train6/weights/best.pt')

result = model.predict('test.jpg')
prediction = model.predict('test.jpg', conf=0.7)

for res in prediction:

    
    res.show()

# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Current device:", torch.cuda.current_device())
#     print("Device name:", torch.cuda.get_device_name(0))