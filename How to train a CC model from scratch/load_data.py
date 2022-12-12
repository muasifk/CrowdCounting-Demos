

import numpy as np
import os
# import sys
# from pathlib import Path
# import re
from glob import glob
# import shutil
# import random
from sklearn.model_selection import train_test_split

def load_data(dataset_path):
    '''
    Load mall dataset
    '''
    img_paths   = sorted(glob(dataset_path + '/frames/*.jpg'))    # List of paths, See sample print(len(filepaths))
    gt_paths    = sorted(glob(dataset_path + '/ground-truth/*.mat'))    # List of paths, See sample print(len(filepaths))
    img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
    gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))

    # Train/Test split
    train_img_paths, val_img_paths, train_gt_paths, val_gt_paths = train_test_split(img_paths, gt_paths, test_size=0.3, random_state=42)
    
    
    
    print('\33[32m')
    print(f'>>>>>>>> Dataset is successfuly loaded .. \N{grinning face}')
    print('\33[36m')
    print('=========================================')
    print(f'Train data (img/gt)   :  {len(train_img_paths)} = {len(train_gt_paths)}')
    print(f'Test data (img/gt)    :  {len(val_img_paths)} = {len(val_gt_paths)}')
    print(f'Total data (img/gt)   :  {len(train_img_paths) + len(val_img_paths)} = {len(train_gt_paths) + len(val_gt_paths)}')
    print('==========================================')                                                                             
    # else:
    #     print(f' \33[91m \N{crying face} >>>>>>>> {ds_name} dataset loading failed .. ') # 
    return train_img_paths, train_gt_paths, val_img_paths, val_gt_paths