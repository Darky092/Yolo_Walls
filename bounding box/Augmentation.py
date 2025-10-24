import numpy
import albumentations as A
import os
import cv2
import tqdm
from CreateFrames import get_first_entry_or_none


def Read_Label(path):
    res = []
    with open(path, 'r') as file:
        for line in file.readlines():
            line = line.rstrip('\n')
            line = line.split()
            add_line = line[1:]
            add_line.append(line[0])
            res.append(add_line)
    return res



def Read_Boxes(boxes):
    res = []
    for el in boxes:
        append_res = el[:-1]
        append_res.insert(0, el[-1])
        res.append(append_res)
    return res


def Augmentation():
    if (get_first_entry_or_none("frames and labels/labels/") is not None):     
        return 'folder is not empty' 


    transform = A.Compose(
        [
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomCrop(width= 600, height=600, p=1),
            A.Rotate(limit=40, p=0.3),
            A.ShiftScaleRotate(p=0.3)
        ], p=0.1),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.5),
        ], bbox_params=A.BboxParams(format='yolo')
    )


    frames = [os.path.join('original frames and labels/original frames/', x) for x in os.listdir('original frames and labels/original frames/')]
    

    for frame in frames:
        for i in range(50):
            bboxes = Read_Label("original frames and labels/labels/"+ frame[:-4].split("/")[-1]+".txt")
            bboxes_converted = []
            for p in bboxes:
                p = list(map(lambda x: float(x), p))
                p[4] = int(p[4])
                bboxes_converted.append(p)
            trans_img = transform(image=cv2.imread(frame), bboxes=bboxes_converted)
            cv2.imwrite("frames and labels/frames/"+frame[:-4].split("/")[-1] + f"_{i}.jpg", trans_img["image"])
        
            with open("frames and labels/labels/" + frame[:-4].split("/")[-1] + f"_{i}.txt", 'w') as f:
                f.write("\n".join(list(map(lambda x: " ".join(map(lambda y: str(y), x)), Read_Boxes(list(map(lambda x: list(x), trans_img["bboxes"])))))))

Augmentation()