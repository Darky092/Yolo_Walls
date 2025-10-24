import shutil
from sklearn.model_selection import train_test_split
import os
from CreateFrames import get_first_entry_or_none

def Split_Dataset():
    if (get_first_entry_or_none("test/labels/") is not None):     
        return 'folder is not empty' 

    RANDOM_STATE = 42


    frames = [os.path.join('frames and labels/frames', x) for x in os.listdir('frames and labels/frames')]
    labels = [os.path.join('frames and labels/labels', x) for x in os.listdir('frames and labels/labels') if x[-3:] == "txt"]


    frames.sort()
    labels.sort()


    train_frames, val_frames, train_labels, val_labels = train_test_split(frames, labels, test_size=0.2, random_state=RANDOM_STATE)
    val_frames, test_frames, val_labels, test_labels = train_test_split(val_frames, val_labels, test_size=0.5, random_state=RANDOM_STATE)


    def move_files_to_folder(list_of_files, destination_folder):
        for f in list_of_files:
            try:
                shutil.copy(f, destination_folder)
            except:
                print(f)
                assert False


    move_files_to_folder(train_frames, 'train/frames/')
    move_files_to_folder(val_frames, 'validation/frames/')
    move_files_to_folder(test_frames, 'test/frames/')
    move_files_to_folder(train_labels, 'train/labels/')
    move_files_to_folder(val_labels, 'validation/labels/')
    move_files_to_folder(test_labels, 'test/labels/')