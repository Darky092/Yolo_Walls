from CreateDirectories import Create_Dir
from CreateFrames import Create_frames
from Augmentation import Augmentation
from Split_Dataset import Split_Dataset
from Training import Training   
from ultralytics import YOLO

if __name__ == '__main__':
    Create_Dir()
    Create_frames('video recording/pills.MOV', 'original frames and labels/original frames')
    Augmentation()
    Split_Dataset()
    Training()

