# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:52:54 2021
CNN Help:  https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

@author: Utility functions
"""
import numpy as np
import torch
import torch.nn as nn


class MCNN(nn.Module):
    '''
    MCNN: Implementation of Multi-column CNN for crowd counting, CVPR 2016
    '''
    def __init__(self, load_weights=False):
        super(MCNN,self).__init__()

        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0)) # 30 is C_out = 12+10+8

        if not load_weights:
            self._initialize_weights()

    def forward(self,x):
        x1=self.branch1(x)
        x2=self.branch2(x)
        x3=self.branch3(x)
        y=torch.cat((x1,x2,x3),1)
        y=self.fuse(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# test code
# if __name__=="__main__":
#     img=torch.rand((1,3,800,1200), dtype=torch.float) # create a random image
#     mcnn=MCNN()
#     out_dmap=mcnn(img)
#     print(out_dmap.shape)



##===================================
##  TF 2
##===================================

# def MCNN_Tf(input_shape):
#     '''
#     Multi-Column CNN Model
#     https://github.com/CommissarMa/Crowd_counting_from_scratch/blob/master/crowd_model/mcnn_model.py
#     https://github.com/svishwa/crowdcount-mcnn/blob/master/src/models.py
#     '''

#     inputs = Input(shape=input_shape)

#     # column 1
#     column_1 = Conv2D(16, (9, 9), padding='same', activation='relu')(inputs)
#     column_1 = MaxPooling2D(2)(column_1)
#     column_1 = (column_1)
#     column_1 = Conv2D(32, (7, 7), padding='same', activation='relu')(column_1)
#     column_1 = MaxPooling2D(2)(column_1)
#     column_1 = Conv2D(16, (7, 7), padding='same', activation='relu')(column_1)
#     column_1 = Conv2D(8, (7, 7), padding='same', activation='relu')(column_1)
    
#     # column 2
#     column_2 = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
#     column_2 = MaxPooling2D(2)(column_2)
#     column_2 = (column_2)
#     column_2 = Conv2D(40, (5, 5), padding='same', activation='relu')(column_2)
#     column_2 = MaxPooling2D(2)(column_2)
#     column_2 = Conv2D(20, (5, 5), padding='same', activation='relu')(column_2)
#     column_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(column_2)
    
#     # column 3
#     column_3 = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs)
#     column_3 = MaxPooling2D(2)(column_3)
#     column_3 = (column_3)
#     column_3 = Conv2D(48, (3, 3), padding='same', activation='relu')(column_3)
#     column_3 = MaxPooling2D(2)(column_3)
#     column_3 = Conv2D(24, (3, 3), padding='same', activation='relu')(column_3)
#     column_3 = Conv2D(12, (3, 3), padding='same', activation='relu')(column_3)
    
    
#     # merge feature map of 3 columns in last dimension
#     merges = Concatenate(axis=-1)([column_1, column_2, column_3])

#     # density map
#     density_map = Conv2D(1, (1, 1), padding='same')(merges) # activation='sigmoid'
#     model = Model(inputs=inputs, outputs=density_map)
    
#     # opt   = keras.optimizers.SGD()
#     # opt = SGD(learning_rate=0.01, momentum=0.9, decay=0.01)
#     opt   = keras.optimizers.Adam() # learning_rate=0.001
#     model.compile(optimizer=opt, loss='mae', metrics='mae')  # metrics=['mse','mae'], metric_mae_count
#     return model

