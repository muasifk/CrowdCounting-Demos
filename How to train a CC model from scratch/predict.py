

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import torch
import torchvision.transforms as transforms
# import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
# import albumentations as A
from torchvision.transforms import ToTensor, Lambda, Compose
from tqdm.notebook import tqdm
from torch import nn

# from skimage.metrics import structural_similarity as ssim
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from torchmetrics.functional import PeakSignalNoiseRatio as psnr
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio





def predict(model, dataloader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # set model to evaluate model
    model.eval()
    test_images = torch.tensor([], device=device)
    test_gt     = torch.tensor([], device=device)
    test_et     = torch.tensor([], device=device)
    
    metric_mse  = 0.0
    metric_mae  = 0.0
    metric_game = 0.0
    metric_ssim = 0.0
    metric_psnr = 0.0
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        # for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader)/dataloader.batch_size)):
        for i, data in enumerate(dataloader):
            batch_error = 0.0
            img, gt = data[0].to(device), data[1].to(device)  # Read a single batch
            et      = model(img)  # predict the output

            metric_mse += (gt.sum() - et.sum()) ** 2
            metric_mae += abs(gt.sum() - et.sum())
            ssim = StructuralSimilarityIndexMeasure().to(device)
            psnr = PeakSignalNoiseRatio().to(device)
            metric_ssim += ssim(et, gt)
            metric_psnr += psnr(et, gt)
            

            ###### GAME
            L = 4
            p = pow(2, L)
            b, c, h, w = gt.shape
            
            for i in range(p):
                for j in range(p):
                    assert et.shape == gt.shape
                    gt_block = gt[:,:, i*h//p:(i+1)*h//p, j*w//p:(j+1)*w//p]
                    et_block = et[:,:, i*h//p:(i+1)*h//p, j*w//p:(j+1)*w//p]
                    metric_game += abs(gt_block.sum() - et_block.sum())


            ### Output to return
            test_images  = torch.cat((test_images, img), 0)  # concatenate images in this batch to 'images'
            test_gt      = torch.cat((test_gt, gt), 0)
            test_et      = torch.cat((test_et, et), 0)
            
        # Calc MSE,  MAE,  GAME
        metric_mse     = metric_mse/len(dataloader.dataset) # find average over all batches
        metric_mae     = metric_mae/len(dataloader.dataset)
        metric_game    = metric_game/len(dataloader.dataset)
        metric_ssim    = metric_ssim/len(dataloader.dataset)
        metric_psnr    = metric_psnr/len(dataloader.dataset)
        
        
        print(f'MSE: {metric_mse}')
        print(f'MAE: {metric_mae}')
        print(f'GAME {pow(2,L)}: {metric_game}')
        print(f'SSIM: {metric_ssim}')
        print(f'PSNR: {metric_psnr}')

    return test_images, test_gt, test_et, metric_mae