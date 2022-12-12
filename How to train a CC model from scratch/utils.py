
# import os, random, time
# import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
# import cv2
# import scipy
# from scipy.ndimage import gaussian_filter
# from scipy.io import loadmat
# import xml.etree.ElementTree as ET
# import torch
# import torchvision.transforms as transforms
# import torchvision.transforms as T
# from torchvision.transforms import ToTensor, Lambda, Compose




##### Show sample test image
def display_sample(img, gt):
    '''Display a single sample from dataset '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    gt  = gt.squeeze(0).numpy()
    img = img.permute(1,2,0).numpy()
    ax1.imshow(img) # image
    ax2.imshow(gt, cmap='jet') # GT Local
    ax1.set_title('Original image', fontweight='bold', fontstretch='ultra-expanded')
    ax2.set_title(f'Actual count: {gt.sum():.0f}', fontweight='bold', fontstretch='ultra-expanded')  
    plt.tight_layout()
    # plt.savefig(figures+ f'/IMG_{i-1}.jpg', dpi=300)
    # print('Image displayed .. \N{smiling face with sunglasses}')
    return fig



##### Show sample test image
def display_prediction(img, gt, et):
    ''' Display a single prediction on dataset'''
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    gt  = gt.squeeze(0).numpy()
    et  = et.squeeze(0).detach().cpu().squeeze(0).numpy()
    img = img.permute(1,2,0).numpy()
   
    ax1.imshow(img) # image
    ax1.set_title(f'Actual count: {gt.sum():.0f}', fontsize=16, fontweight='medium', fontstretch='ultra-expanded')
    ax2.imshow(gt, cmap='jet') # image
    ax2.set_title(f'Actual count: {gt.sum():.0f}', fontsize=16, fontweight='medium', fontstretch='ultra-expanded')  
    ax3.imshow(et, cmap='jet') 
    ax3.set_title(f'Predicted counts: {et.sum():.0f}', fontsize=16, color='red', fontweight='medium', fontstretch='ultra-expanded')
    plt.tight_layout()
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    return fig




def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, val_metric, filename):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch'     :  epoch + 1,
        'train_loss':  train_loss,
        'val_loss'  :  val_loss,
        'val_metric':  val_metric,
        'state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        filename)
    
    
def load_checkpoint(model, optimizer, device, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print(f'Checkpoint exists (Loading..) {filename}')
        checkpoint    = torch.load(filename, map_location=device)
        start_epoch   = checkpoint['epoch']
        train_loss    = checkpoint['train_loss']
        val_loss      = checkpoint['val_loss']
        val_metric    = checkpoint['val_metric']
        # print('check', type(checkpoint['state_dict']))
        if isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            
        # model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'===> loaded checkpoint {filename} epoch {start_epoch} ... MAE: {val_metric:.2f}')
    else:
        print("===> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, val_metric
    
