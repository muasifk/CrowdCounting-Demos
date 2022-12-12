

import os
import time
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



def trainer(parameters):
    '''
    My Custom method to train and validate a model
    '''
    ## Parameters
    model         = parameters['model']
    train_dl      = parameters['train_dl']
    val_dl        = parameters['val_dl']
    optimizer     = parameters['optimizer']
    criterion     = parameters['criterion']
    # logdir        = parameters['logdir']
    lr_scheduler  = parameters['lr_scheduler']
    device        = parameters['device']
    n_epochs      = parameters['n_epochs']
    checkpoint    = parameters['checkpoint']
    
    val_metric_prev  = parameters['val_metric']
    val_loss_prev    = np.Inf  # min val loss
    
    ## Returns
    epochs     = []
    train_loss = []
    val_loss   = []
    val_metric = []
    
    
    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/

    # tb = SummaryWriter()
    
    ###############################################
    #   Start Training
    ###############################################
    print('You are using GPU: ', torch.cuda.get_device_name(device)) if torch.cuda.is_available() else print('You are using CPU')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  updated')
    
    for epoch in range(1, n_epochs+1):
        print(f"Epoch {epoch} of {n_epochs}")
        
        
        ## ===========================================================
        ##  Training Loop
        ## ===========================================================
        model.train()
        train_loss_epoch = 0.0
        # for i, data in tqdm(enumerate(train_dl), total=int(len(train_dl)/train_dl.batch_size), desc='Training epoch'):
        for i, data in enumerate(train_dl):
            img, gt  = data[0].to(device), data[1].to(device) # Read a single batch
            optimizer.zero_grad()  # sets gradients to zeros
            et = model(img) # predict the outputs (inputs is batch of images)
            batch_loss  = criterion(gt, et) # calculate loss (scalar value: mean or sum of losses for all images in the batch)
            train_loss_epoch += batch_loss.item() # add batch_loss to find cumulative epoch_loss which will be averaged later
            batch_loss.backward()  # Backpropagation
            optimizer.step()
            time.sleep(0.01)
        train_loss_epoch = train_loss_epoch/len(train_dl.dataset) # average over the number of images to get mean error for thw whole epoch
        
        
        
        ## ===========================================================
        ##  Validation Loop
        ## ===========================================================
        model.eval()
        val_loss_epoch   = 0.0
        val_metric_epoch = 0.0
    
        with torch.no_grad():
            # for i, data in tqdm(enumerate(val_dl), total=int(len(val_dl)/val_dl.batch_size), desc='Validation epoch'):
            for i, data in enumerate(val_dl):
                img, gt = data[0].to(device), data[1].to(device) # Read a single batch
                et    = model(img) # predict the output
                batch_loss  = criterion(gt, et) # calculate loss (scalar value: mean or sum of losses for all images in the batch)
                val_loss_epoch += batch_loss.item() # add batch_loss to find cumulative epoch_loss which will be averaged later
                val_metric_epoch += abs(gt.sum() - et.sum())         
                time.sleep(0.01)
            val_loss_epoch     = val_loss_epoch/len(val_dl.dataset) # find average over all batches
            val_metric_epoch   = val_metric_epoch/len(val_dl.dataset) # find average over all batches
            
            
                          
        ## Update learning rate
        if lr_scheduler is not None:
            # lr_scheduler.step() # Update learning rate (for ReduceLROnPlateau)  val_loss_epoch
            lr_scheduler.step(val_loss_epoch) # Update learning rate (for CyclicLR)

        ## Record losses
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        val_metric.append(val_metric_epoch)
        
        
        ## ===========================================================
        ##  STDOUT and checkpointing
        ## ===========================================================
        if lr_scheduler is not None:
            print(f'Learning Rate: {lr_scheduler.get_lr()[0] :.5f}')
        print(f'Epoch:{epoch}  ==> \
        Train/Valid Loss: {train_loss_epoch:.4f} / {val_loss_epoch:.4f} ... MAE={val_metric_epoch:.2f}')
        
        # Save checkpoint
        if checkpoint is not None:
            if (val_metric_epoch < val_metric_prev):# or (val_loss_epoch < val_loss_prev):
                print(f'Validation MAE decreased ({val_metric_prev:.2f} --> {val_metric_epoch:.2f}):  Saving model ...')
                checkpoints = {'epoch'     : epoch, 
                              'train_loss' : train_loss_epoch, 
                              'val_loss'   : val_loss_epoch, 
                              'val_metric' : val_metric_epoch, 
                               'state_dict' : model.state_dict(), 
                              'optimizer_state_dict': optimizer.state_dict()}
                
                if isinstance(model, nn.DataParallel):
                    'Converting to single GPU before saving...'
                    checkpoints['state_dict'] : model.module.state_dict() 
                    
                torch.save(checkpoints, checkpoint)
                
                # Update loss and MAE to compate in next epoch
                val_loss_prev   = val_loss_epoch
                val_metric_prev = val_metric_epoch
        
        epochs.append(epoch)
       
    ## Return a "history" dictionary of lists of "epochs, losses, metric values"
    history = {'epochs'      : epochs,
               'train_loss'  : train_loss,
               'val_loss'    : val_loss,
               'val_metric'  : val_metric}
        
    if epoch == n_epochs:
        print('Training completed .. \n')
    
    # Close tensorboard
    # tb.close()
    
    return history
