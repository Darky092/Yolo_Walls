import albumentations as A
import os
import numpy as np
import cv2


def ReadLabels(path: str) -> list[tuple]:
    with open(path, 'r') as file:
        res = file.readlines()
        yoloLabelsData = []


        for data in res:
            data = data.strip()
            data = data.strip('\n')
            data = data.split()
            yoloLabelsData.append((int(data[0]),list(map(float, data[1:]))))

        
        return yoloLabelsData


def FromYoloCordConvertion(yoloCord: list[float], imageHight: float, imageWidth: float) ->list[tuple]:
    cords = []
    for i in range(0, len(yoloCord), 2):
        x = int(yoloCord[i] * imageWidth)
        y = int(yoloCord[i+1] * imageHight)
        cords.append((x,y))
    return cords


def InYoloCordConvertion(cord: list[tuple], imageHight: float, imageWidht: float) ->list[float]:
    yoloCords = []
    for (x,y) in cord:
        norm_x = float(x / imageWidht)
        norm_y = float(y/ imageHight)
        norm_x = np.clip(norm_x, 0.0, 1.0)
        norm_y = np.clip(norm_y, 0.0, 1.0)
        yoloCords.append(norm_x)
        yoloCords.append(norm_y)
    return yoloCords


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),  
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.2,
        p=0.4
    ),
    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=15,
        val_shift_limit=15,
        p=0.3
    ),
    A.GaussNoise(var_limit=(1.0, 4.0), p=0.1),  
    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.2),  
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        

for _ in range(23):
    for frame in os.listdir('origin frames and labels/origin frames'):
        
        image = cv2.imread(filename=f'origin frames and labels/origin frames/{frame}')
        poligonPath = f'origin frames and labels/origin labels/{frame[:-4]}' + '.txt'
        h , w = image.shape[:2]
        classPlusCords = ReadLabels(poligonPath)
        oldYoloCords = []
        for x in classPlusCords:
            oldYoloCords.append(x[1])

        
        oldNormCords = []
        lenObjects = []
        for x in oldYoloCords:
            cords = FromYoloCordConvertion(yoloCord=x,imageHight=h,imageWidth=w)
            lenObjects.append(len(cords)* 2)
            oldNormCords.extend(cords)

        transformed = transform(image=image,keypoints=oldNormCords)
        newImage = transformed['image']
        n_h, n_w = newImage.shape[:2]
        newNormCords = transformed['keypoints']
        newYoloCords = InYoloCordConvertion(cord=newNormCords,imageHight=n_h,imageWidht=n_w)
        decidedYoloCords = []
        for x in lenObjects:
            obj = []
            for i in range(x):
                obj.append(newYoloCords[i])
                
            decidedYoloCords.append(obj)

            newYoloCords = [item for i, item in enumerate(newYoloCords) if i >= x ]

        for i in range(0, len(classPlusCords)):
            classPlusCords[i] = (classPlusCords[i][0],decidedYoloCords[i])
        classPlusCordsString = ''
        for _tuple in classPlusCords:
            classPlusCordsString += str(_tuple[0]) + ' ' + ' '.join(list(map(str, _tuple[1]))) + '\n'
        classPlusCordsString = classPlusCordsString.strip('\n')

        cv2.imwrite(filename=f'frames and labels/frames/{_}_{frame}',img=newImage)
        with open(f'frames and labels/labels/{_}_{frame[:-4]}.txt','w') as file:
            file.write(classPlusCordsString)




print('augmentation has been done')