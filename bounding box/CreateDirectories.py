from os import mkdir


def Create_Dir():
    for directory_name in ['video recording','original frames and labels',
                           'original frames and labels/original frames',
                           'original frames and labels/labels',
                           'frames and labels',
                           'frames and labels/frames',
                           'frames and labels/labels',
                           'train', 'test', 'validation',
                           'train/images', 'test/images', 'validation/images', 
                           'train/labels', 'test/labels', 'validation/labels']:
        try:
            mkdir(directory_name)
            print(f'directory {directory_name} has been created')
        except:
            print(f'directory {directory_name} alredy created')
