# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:52:54 2021

@author: Utility functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
# import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# import albumentations as A
from torchvision.transforms import ToTensor, Lambda, Compose
from tqdm.notebook import tqdm
from torch import nn
import time

'''
Help
https://www.kaggle.com/tthien/shanghaitech-a-test-density-gen
https://github.com/CommissarMa/MCNN-pytorch/blob/master/my_dataloader.py

MAE = sqrt(2/pi)*sqrt(MSE)  ## Holds true when bias is zero or when errors follow normal distribution with zero mean, and constant variance
'''


class CrowdDataset(Dataset):
    '''
    Custom dataset using Dataset API
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    
    Three functions:
    __init__      run once when instantiating the Dataset object
    __len__       returns the number of samples in our dataset.
    __getitem__   loads and returns a sample from the dataset at the given index idx
    
    
    Return:
    img_tensor:    of shape (channels, height, width) e.g., (3,384,512)
    gt_tensor:     of shape (channels, height, width) e.g., (1,96,128)
    
    '''
    def __init__(self, img_paths, gt_paths, img_downsample, gt_downsample, sigma, augmentation): # ds_name
        self.img_names       = img_paths
        self.gt_names        = gt_paths
        self.img_downsample  = img_downsample
        self.gt_downsample   = gt_downsample
        # self.ds_name         = ds_name
        self.augmentation    = augmentation
        self.sigma           = sigma
        self.n_samples       = len(self.img_names)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
    
        ##############  1. Read image
        img_name = self.img_names[index]
        img  = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)  # when img_paths are provided
        # print('Original image', img.shape)
        
        ###############  1. Reading GTs
        gt_name  = self.gt_names[index]
        
#         if self.ds_name == 'CARPK':
#             mat      = loadmat(gt_name)   # when gt_path is provided
#             pos      = mat.get("annot")
            
#         if self.ds_name == 'DroneRGBT':
#             tree = ET.parse(gt_name)
#             root = tree.getroot()
#             pos  = [ [int(x.text), int(y.text)] for x,y in zip(root.iter('x'), root.iter('y'))] ## x is width, y is height
            
#         if self.ds_name == 'ShanghaiTechPartA' or self.ds_name == 'ShanghaiTechPartB':
#             mat    = loadmat(gt_name)   # when gt_path is provided
#             pos    = mat.get("image_info")[0][0][0][0][0]
            
#         if self.ds_name == 'UCF-QNRF':
#             mat    = loadmat(gt_name)   # when gt_path is provided
#             pos    = mat.get("annPoints")
            
        # if self.ds_name == 'Mall':
        mat  = loadmat(gt_name)
        pos  = mat['annot']
        count = mat['count'][0][0]
        assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
        
#         if self.ds_name == 'UCF_CC_50':
#             mat  = loadmat(gt_name)
#             pos  = mat['annPoints']
#             count = len(pos)
#             assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
            
#         if self.ds_name == 'JHU-Crowd':
#             txt = np.loadtxt(gt_name)
#             pos = txt[:,0:2]
#             count = len(pos)
#             assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
            
        ##############  2. Create density maps
        fixed_kernel    = True
        adaptive_kernel = False
        # print('processing now image', img_name)
        z    = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for i, j in pos:
            try:
                z[int(j), int(i)] = 1 # Transformation of coordinates
            except:
                pass
        if fixed_kernel is True:
            gt = gaussian_filter(z, self.sigma)
        
        if adaptive_kernel is True:
            k = 3 # select value
            dm   = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  
            tree = scipy.spatial.KDTree(pos, leafsize=2)   # from sklearn.neighbors import KDTree, leaf_size measures speed
            dist, ind = tree.query(pos, k=k+1)   # Distance to k+1 closest poinst (first point is self)
            for i in range( len(pos) ):
                sigma = np.mean(dist[i, 1:(k+1)]) / k # average of three distances
                sigma = sigma / 2 # half of average distance to k neighbors
            gt = gaussian_filter(z, sigma=sigma) # Which one is correct? above inside for loop or outside as this line?
       

        ##############  3. Downsample images
        if self.img_downsample>1: # to downsample image and density-map to match deep-model.
            ds_rows = int(img.shape[0]//self.img_downsample) # downsample rows
            ds_cols = int(img.shape[1]//self.img_downsample) # downsample columns
            img     = cv2.resize(img, (ds_cols, ds_rows))
            
            
        ############## 4. Downsample density maps
        if self.gt_downsample>1:
            ds_rows = int(img.shape[0]//self.gt_downsample)  # // 9/2=4.5 ,  9//2=4
            ds_cols = int(img.shape[1]//self.gt_downsample)
            gt      = cv2.resize(gt, (ds_cols,ds_rows))
            gt      = gt[np.newaxis,:,:] *self.gt_downsample*self.gt_downsample
        gt_tensor   = torch.tensor(gt, dtype=torch.float)
        
        
        
            
        ##############  5. Augmentation
        # Normalize first
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean=0, std=1)])
        img_tensor = normalize(img) # normalize first
        
        ## Choose augmentations
        h_flip     = T.RandomHorizontalFlip(p=0.5)
        c_jitter   = T.ColorJitter(brightness=0.3, contrast=0, saturation=0.2, hue=0.2) # brightness=0.3, contrast=0, saturation=0, hue=0.2
        rand_sharp = T.RandomAdjustSharpness(sharpness_factor=2)
        rand_equal = T.RandomEqualize()
    
        ## Apply augmentation
        if self.augmentation is True: # img is converted to tensor by transform
            img_tensor = c_jitter(img_tensor) # jitter
            # img_tensor = TF.adjust_brightness(img_tensor, torch.rand(1)*2) # 0=black, 1=original, 2= double brightness
            # img_tensor = TF.adjust_hue(img_tensor, torch.distributions.uniform.Uniform(-0.5,0.5) ) # 0=original, -0.5, 0.5
            ## Horizontal flip both img and gt
            p = torch.rand(1)
            if p > 0.5:
                img_tensor = TF.hflip(img_tensor)
                gt_tensor  = TF.hflip(gt_tensor)
            
            
            
        ## count
        gt_count = int(gt_tensor.sum())
        return img_tensor, gt_tensor
    


    
# ###############################################################
# Takes img_size and gt_size as argument
# ###############################################################   

class CrowdDataset2(Dataset):
    '''
    Custom dataset using Dataset API
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    
    Three functions:
    __init__      run once when instantiating the Dataset object
    __len__       returns the number of samples in our dataset.
    __getitem__   loads and returns a sample from the dataset at the given index idx
    
    
    Return:
    img_tensor:    of shape (channels, height, width) e.g., (3,384,512)
    gt_tensor:     of shape (channels, height, width) e.g., (1,96,128)
    
    '''
    def __init__(self, img_paths, gt_paths, img_size, gt_size, ds_name, sigma, augmentation):
        self.img_names       = img_paths
        self.gt_names        = gt_paths
        self.img_size        = img_size
        self.gt_size         = gt_size
        self.ds_name         = ds_name
        self.augmentation    = augmentation
        self.sigma           = sigma
        self.n_samples       = len(self.img_names)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        # print(f'Reading sample: {self.img_names[index]}')
        ##############  1. Read image
        img_name = self.img_names[index]
        img  = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)  # when img_paths are provided
        # print('Original image', img.shape)
        
        ###############  1. Reading GTs
        gt_name  = self.gt_names[index]
        
        if self.ds_name == 'CARPK':
            mat      = loadmat(gt_name)   # when gt_path is provided
            pos      = mat.get("annot")
            
        if self.ds_name == 'DroneRGBT':
            tree = ET.parse(gt_name)
            root = tree.getroot()
            pos  = [ [int(x.text), int(y.text)] for x,y in zip(root.iter('x'), root.iter('y'))] ## x is width, y is height
            
        if self.ds_name == 'ShanghaiTechPartA' or self.ds_name == 'ShanghaiTechPartB':
            mat    = loadmat(gt_name)   # when gt_path is provided
            pos    = mat.get("image_info")[0][0][0][0][0]
            
        if self.ds_name == 'UCF-QNRF':
            mat    = loadmat(gt_name)   # when gt_path is provided
            pos    = mat.get("annPoints")
            
        if self.ds_name == 'Mall':
            mat  = loadmat(gt_name)
            pos  = mat['annot']
            count = mat['count'][0][0]
            assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
        
        if self.ds_name == 'UCF_CC_50':
            mat  = loadmat(gt_name)
            pos  = mat['annPoints']
            count = len(pos)
            assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
            
        if self.ds_name == 'JHU-Crowd':
            txt = np.loadtxt(gt_name, delimiter = " ")
            # print(f'Index {index} ==> {self.img_names[index]}')
            if txt.ndim == 1:
                print('POS has single dimension')
                pass
            else:
                pos = txt[:,0:2]
            # df = pd.read_csv(gt_name, sep=" ", header=None)
            # pos = df.iloc[:,0:2]
            # pos = pos.values.tolist()
            # count = len(pos)
            # assert count == len(pos) , f"Error in reading correct file: Mismatch in [count != len(pos)]"
     
        # if self.ds_name == 'Lusail':
        #     df = pd.read_csv(gt_name)
        #     df = df.iloc[:,5]
        #     x = df.str.split(',', expand=True)[1].str.split(':', expand=True)[1]
        #     y = df.str.split(',', expand=True)[2].str.split(':', expand=True)[1].str.replace(r'\D', '')
        #     x = x.astype(int)
        #     y = y.astype(int)
        #     pos = list(zip(x, y))

            
        ##############  2. Create density maps
        fixed_kernel    = True
        adaptive_kernel = False
        # print('processing now image', img_name)
        z    = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for i, j in pos:
            try:
                z[int(j), int(i)] = 1 # Transformation of coordinates
            except:
                pass
            
        if fixed_kernel is True:
            gt = gaussian_filter(z, self.sigma)
        
        if adaptive_kernel is True:
            k = 3 # select value
            dm   = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  
            tree = scipy.spatial.KDTree(pos, leafsize=2)   # from sklearn.neighbors import KDTree, leaf_size measures speed
            dist, ind = tree.query(pos, k=k+1)   # Distance to k+1 closest poinst (first point is self)
            for i in range( len(pos) ):
                sigma = np.mean(dist[i, 1:(k+1)]) / k # average of three distances
                sigma = sigma / 2 # half of average distance to k neighbors
            gt = gaussian_filter(z, sigma=sigma) # Which one is correct? above inside for loop or outside as this line?
       
        ############### 3. Downsample images and density maps
        if self.img_size is not None:
            # img     = cv2.resize(img, (self.img_size[2], self.img_size[1]))
            img     = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            # print('New image', img.shape)
            
        if self.gt_size is not None:
            RH, RW  = gt.shape[0]/self.gt_size[0], gt.shape[1]/self.gt_size[1]
            # print('RW, RW', RH, RW)
            gt      = cv2.resize(gt, (self.gt_size[1], self.gt_size[0]))
            # print('new gt', gt.shape, gt.sum())
            gt      = gt[np.newaxis,:,:]* RW * RH            
        gt_tensor   = torch.tensor(gt, dtype=torch.float)
        # print('gt_tensor', gt_tensor.shape, gt_tensor.sum())
            
        ##############  4. Augmentation
        # Normalize first
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean=0, std=1)])
        img_tensor = normalize(img) # normalize first
        
        ## Choose augmentations
        h_flip     = T.RandomHorizontalFlip(p=0.5)
        c_jitter   = T.ColorJitter(brightness=0.3, contrast=0, saturation=0.2, hue=0.2) # brightness=0.3, contrast=0, saturation=0, hue=0.2
        rand_sharp = T.RandomAdjustSharpness(sharpness_factor=2)
        rand_equal = T.RandomEqualize()
    
        ## Apply augmentation
        if self.augmentation is True: # img is converted to tensor by transform
            img_tensor = c_jitter(img_tensor) # jitter
            # img_tensor = TF.adjust_brightness(img_tensor, torch.rand(1)*2) # 0=black, 1=original, 2= double brightness
            # img_tensor = TF.adjust_hue(img_tensor, torch.distributions.uniform.Uniform(-0.5,0.5) ) # 0=original, -0.5, 0.5
            ## Horizontal flip both img and gt
            p = torch.rand(1)
            if p > 0.5:
                img_tensor = TF.hflip(img_tensor)
                gt_tensor  = TF.hflip(gt_tensor)
            
            
            
        ## count
        gt_count = int(gt_tensor.sum())
        return img_tensor, gt_tensor
    
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


# ###############################################################
# # Old
# ###############################################################

# class CrowdDataset(Dataset):
#     '''
#     Custom dataset using Dataset API
#     https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    
#     Three functions:
#     __init__      run once when instantiating the Dataset object
#     __len__       returns the number of samples in our dataset.
#     __getitem__   loads and returns a sample from the dataset at the given index idx
    
    
#     Return:
#     img_tensor:    of shape (channels, height, width) e.g., (3,384,512)
#     gt_tensor:     of shape (channels, height, width) e.g., (1,96,128)
    
#     '''
#     def __init__(self, img_root, gt_root, img_downsample, gt_downsample, transform):
#         # self.img_root       = img_root
#         # self.gt_root        = gt_root
#         self.img_names       = img_root
#         self.gt_name         = gt_root
#         self.img_downsample  = img_downsample
#         self.gt_downsample   = gt_downsample
#         self.transform       = transform

        
#         # Read all images names
#         # self.img_names    = [filename for filename in os.listdir(img_root) if os.path.isfile(os.path.join(img_root,filename))]
#         # self.gt_names     = [filename for filename in os.listdir(gt_root) if os.path.isfile(os.path.join(gt_root,filename))]
#         self.n_samples     = len(self.img_names)
        
#     def __len__(self):
#         return self.n_samples
    
#     def __getitem__(self,index):
#         assert index <= len(self), 'index range error'
    
#         ##############  1. Read image
#         img_name = self.img_names[index]
#         #print('Reading', img_name)
#         # img  = cv2.cvtColor(cv2.imread(os.path.join(self.img_root, img_name)), cv2.COLOR_BGR2RGB) # when img_dir is provided
#         img  = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)  # when img_paths are provided
#         # img  = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # float32 [0,1] # Do not normalize at this stage
#         # print('Original image', img.shape)
        
#         ##############  2. Create density maps
#         fixed_kernel    = True
#         adaptive_kernel = False
        
#         # gt_name  = self.gt_names[index]
#         # mat      = loadmat(os.path.join(self.gt_root, gt_name)) # when gt_dir is provided
        
#         ## When reading positions data from the original "mall_gt" file
#         # gt_name  = self.gt_name
#         # mat  = loadmat(gt_name)
#         # pos  = mat['frame'][0][index][0][0][0]       
        
#         ## When using the pre-generated GT (.mat) files
#         gt_name  = self.gt_name[index]
    
#         mat  = loadmat(gt_name)
#         pos  = mat['annot']
#         count = mat['count'][0][0]
#         assert count == len(pos) , f"Error in reading corect file: Mismatch in [count != len(pos)]"
        
        
#         z    = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
#         for i, j in pos:
#             z[int(j), int(i)] = 1 # Transformation of coordinates
            
#         if fixed_kernel is True:
#             sigma = 10 # select value
#             gt = gaussian_filter(z, sigma)
        
#         if adaptive_kernel is True:
#             k = 3 # select value
#             dm   = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  
#             tree = scipy.spatial.KDTree(pos, leafsize=2)   # from sklearn.neighbors import KDTree, leaf_size measures speed
#             dist, ind = tree.query(pos, k=k+1)   # Distance to k+1 closest poinst (first point is self)
#             for i in range( len(pos) ):
#                 sigma = np.mean(dist[i, 1:(k+1)]) / k # average of three distances
#                 sigma = sigma / 2 # half of average distance to k neighbors
#             gt = gaussian_filter(z, sigma=sigma) # Which one is correct? above inside for loop or outside as this line?
            
#         ###############  2. Load pre-generated density maps
#         # gt_name  = self.gt_names[index]
#         # gt       = np.load(gt_name)  # when path is provided
        
#         # gt_dmap = np.load(os.path.join(self.gt_root, gt_name))  # when filenames are provided
#         # gt_dmap = np.load(os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.npy'))) # one .npy file
        
#         ############### 5. Downsample density maps
#         if self.gt_downsample>1:
#             ds_rows = int(img.shape[0]//self.gt_downsample)  # // 9/2=4.5 ,  9//2=4
#             ds_cols = int(img.shape[1]//self.gt_downsample)
#             gt      = cv2.resize(gt,(ds_cols,ds_rows))
#             gt      = gt[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample
#         gt_tensor   = torch.tensor(gt, dtype=torch.float)
        
        
#         ##############  3. Downsample images
#         if self.img_downsample>1: # to downsample image and density-map to match deep-model.
#             ds_rows = int(img.shape[0]//self.img_downsample) # downsample rows
#             ds_cols = int(img.shape[1]//self.img_downsample) # downsample columns
#             img     = cv2.resize(img, (ds_cols, ds_rows))
            
            
#         ##############  4. Augmentation
#         if self.transform is not None: # img is converted to tensor by transform
#             img_tensor = self.transform(img)
        
#         if self.transform is None: # 
#             img_tensor  = torch.tensor(img)
#             img_tensor  = img_tensor/255
#             img_tensor  = img_tensor.permute((2,0,1))
        

#         return img_tensor, gt_tensor