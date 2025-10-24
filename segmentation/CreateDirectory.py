import os


def CreateDirectory():
    for dirName in ['origin frames and labels','origin frames and labels/origin frames'
                    ,'origin frames and labels/origin labels','frames and labels','frames and labels/frames',
                    'frames and labels/labels','train','train/images','train/labels','validation','validation/images','validation/labels',
                    'test','test/images','test/labels']:
        try:
            os.mkdir(dirName)
        except Exception as ex:
            print(ex)


