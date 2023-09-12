# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import h5py
from skimage.color import rgb2yuv, yuv2rgb
import matplotlib.image as mpimg

class LoadDataset(Dataset):
    '''
    Loading dataset which only contains Y channel data. Used for loading training data
    NOTE: For Loading RGB test/validation data, used LoadRGBDataset()
    '''
    def __init__(self, datafile, permute=False,N=1000):
        self.data_x,self.data_y  = self.load_data(datafile,N,permute)
        self.x_shape  = self.data_x.shape
        self.y_shape  = self.data_y.shape

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data_x[idx]
        y = self.data_y[idx]
    
        return x,y

    def load_data(self,datafile,N,permute=False):

        assert os.path.exists(datafile), 'Error: The data file ' + datafile + ' is unavailable.'
        
        if datafile.endswith('.npy'):

            data = np.load(datafile).astype(np.float32)
            np.random.shuffle(data)
            
        elif datafile.endswith('.h5'):
            file = h5py.File(datafile, "r+")
            data = np.array(file["/images"][0::])
            file.close()

        totalN,H,W,channels = data.shape
        if N == None:
            n_use = totalN
        else:
            n_use = N
            assert totalN >= n_use
        
        assert channels == 2
        data_x = torch.tensor(data[0:n_use,:,:,0:1])
        data_y = torch.tensor(data[0:n_use,:,:,1:])

        # Hacky permute
        if permute:
            data_x = data_x.permute(0,3,1,2)
            data_y = data_y.permute(0,3,1,2)   


        print(f'     *** Datasets:')
        print(f'         ... samples loaded   = {n_use} of {totalN}')
        print(f'         ... sample dimension = {H}X{W}X{channels}')   


        return data_x,data_y


class LoadRGBDataset(Dataset):
    '''
    Loading dataset which only contains RGB data. 
    NOTE: Do not use for loading dataset needed for training. Use LoadDataset() instead
    '''
    def __init__(self, datafile, permute=False,N=1000):
        self.data_x,self.data_y,self.data_uv  = self.load_data(datafile,N,permute)
        self.x_shape  = self.data_x.shape
        self.y_shape  = self.data_y.shape
            
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data_x[idx]
        y = self.data_y[idx]
        uv = self.data_uv[idx]
        return x,y,uv

    def load_data(self,datafile,N,permute=False):

        # NOTE: If RGB data is provided, need to extract Y channels to be fed to network

        assert os.path.exists(datafile), 'Error: The data file ' + datafile + ' is unavailable.'
        
        if datafile.endswith('.npy'):

            data = np.load(datafile).astype(np.float32)
            totalN,H,W,channels = data.shape
        elif datafile.endswith('.h5'):
            file = h5py.File(datafile, "r+")
            data = np.array(file["/images"][0::])
            file.close()

        totalN,H,W,channels = data.shape    
        if N == None:
            n_use = totalN
        else:
            n_use = N
            assert totalN >= n_use
        assert channels == 6
        data_x  = torch.zeros([n_use,H,W,1])
        data_y  = torch.zeros([n_use,H,W,1])
        data_uv = torch.zeros([n_use,H,W,2])
        for i in range(n_use):
            x_yuv = rgb2yuv(data[i,:,:,0:3])  
            y_yuv = rgb2yuv(data[i,:,:,3:]) 
            data_x[i,:,:,0]  = torch.tensor(x_yuv[:,:,0])
            data_y[i,:,:,0]  = torch.tensor(y_yuv[:,:,0])
            data_uv[i,:,:,0:2] = torch.tensor(y_yuv[:,:,1:3])

        # Hacky permute 
        if permute:
            data_x = data_x.permute(0,3,1,2)
            data_y = data_y.permute(0,3,1,2)
            data_uv = data_uv.permute(0,3,1,2)

        print(f'     *** RGB Dataset:')
        print(f'         ... Converted to YUV color model')
        print(f'         ... samples loaded   = {n_use} of {totalN}')
        print(f'         ... sample dimension = {H}X{W}X{channels}')   


        return data_x,data_y,data_uv        


def LoadRainyImage(fname,psize,scale=True):

    ''' Function to load RGB image and create patches
        INPUT-----
        fname: Name of image file
        psize: Patch size to create. Extra pixels are cropped out.
        scale: Will scale image pixel value to [0,1]^3. Set to true if
               input image pixels take values in [0,255] 
        OUTPUT (working with a cropped image) -----
        number of patches in x and y directions,
        stacked Y channels of each patch,
        stacked UV channels of each patch,
        normalized Y intensity of each patch (maybe used for post-processing de-rained image)      
    '''

    img = mpimg.imread(fname).astype(np.float32)
    H,W,_  = img.shape

    if scale:
        img = img/255.0

    N1 = int(H/psize)
    N2 = int(W/psize)

    cropped_img = img[0:N1*psize,0:N2*psize,:]

    img_yuv = rgb2yuv(cropped_img)

    y_channel_data  = np.zeros((N1*N2,psize,psize,1))
    uv_channel_data  = np.zeros((N1*N2,psize,psize,2))

    y_alpha = np.zeros((N1*N2,1))

    cnt = 0
    for i in range(N1):
        for j in range(N2):
            sample    = img_yuv[i*psize:(i+1)*psize,j*psize:(j+1)*psize,:]
            y_channel_data[i*N1+j]  = sample[:,:,0:1]
            uv_channel_data[i*N1+j] = sample[:,:,1::]

            y_alpha[cnt] = np.mean(sample)
            cnt+=1

    y_beta = y_alpha/np.sum(y_alpha)

    return N1,N2,y_channel_data,uv_channel_data,y_beta       


def syuv2rgb(img):
    img_yuv = np.copy(img)
    #cliping Y channel
    img_yuv[:,:,0] = np.maximum(img_yuv[:,:,0],0.0)
    img_yuv[:,:,0] = np.minimum(img_yuv[:,:,0],1.0)

    img_rgb = yuv2rgb(img_yuv)
    img_rgb = np.maximum(img_rgb,0.0)
    img_rgb = np.minimum(img_rgb,1.0)
    return img_rgb
    








