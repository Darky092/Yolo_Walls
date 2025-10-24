import cv2
from os import scandir

# Проверка пуст ли каталог
def get_first_entry_or_none(path):
    with scandir(path) as it:
        try:
            return next(it)
        except StopIteration:
            return None  

#Нарезка кадров
def Create_frames(videoPath: str, framesPath: str):


    if (get_first_entry_or_none(framesPath) is not None):     
        return 'folder is not empty' 
    

    cap = cv2.VideoCapture(videoPath)
    frameCounter = 0
    frameCount = 1

    
    if (cap.isOpened()== False):
        print("Error opening video file")

    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if frameCounter % 15 == 0:
                cv2.imwrite(f'{framesPath}/frame{frameCount}.png', frame)            
                frameCount += 1             
        else:
            break
        frameCounter += 1
    cap.release()
    return 'frames has been created'


    cv2.destroyAllWindows()
