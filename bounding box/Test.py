import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
# names = model.names
# # font
# font = cv2.FONT_HERSHEY_SIMPLEX

# # org
# org = (60, 50)

# # fontScale
# fontScale = 1
 
# # Blue color in BGR
# color = (255, 0, 0)

# # Line thickness of 2 px
# thickness = 2
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     imageReader = frame
#     count = 1
#     colorx = 0
#     color = (count, 0, 0)
#     prediction = model.predict(frame, conf = 0.6)
#     for result in prediction:
#         classes = result.boxes.cls.numpy()
#         boxes = result.boxes.xyxy.numpy()
#     for cls, box in zip(classes, boxes):
#         imageReader = cv2.rectangle(imageReader,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(count, colorx, 0),5)  
#         c = names[int(cls)]
#         for name in c:
#             imageReader = cv2.putText(imageReader, f'{c}', (int(box[2]+5),int(box[1]+20)), font, fontScale, (count, colorx, 0), thickness, cv2.LINE_AA)
#             count = count + 20
#             colorx += 10
#     cv2.imshow('web' ,imageReader)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()



prediction = model.predict('test.jpg', conf= 0.5)

for res in prediction:

    
    res.show()