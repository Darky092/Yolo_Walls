import albumentations as A
import os
import cv2
import numpy as np


image = cv2.imread('test.jpg')
h, w = image.shape[:2]


with open('test.txt','r') as file:
    res =  file.readline().strip()
    parts = res.split(' ')
    parts = parts[1:]


points = []

transform = A.Compose([
    A.RandomCrop(height=h,width=w),
    A.HorizontalFlip(p=1),
    A.GaussianBlur(p=1)
    ], keypoint_params=A.KeypointParams(format='xy',remove_invisible=False,angle_in_degrees=True))

for i in range(0, len(parts), 2):
    x = int(float(parts[i])* w)
    y = int(float(parts[i+1])* h)
    points.append((x,y))

points = np.array(points, dtype=np.int32)


transformed = transform(image=image,keypoints=points)

transformedImage = transformed['image']
transformedPoints = transformed['keypoints']

n_h, n_w = transformedImage.shape[:2]
yoloCord = []

for (x,y) in transformedPoints:
    norm_x = x / n_w
    norm_y = y / n_h
    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_y = np.clip(norm_y, 0.0, 1.0)
    yoloCord.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

print(yoloCord)

newpoint = np.array([[[int(x), int(y)] for (x,y) in transformedPoints]], dtype=np.int32)


cv2.polylines(img=image,pts=[points],color=(255,0,0), isClosed=True, thickness=1)
cv2.imshow(winname='test', mat= image)
cv2.waitKey(0)
cv2.destroyAllWindows()   


