# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
from torch import cat,rsqrt,rand,randn,autograd,ones_like,norm,pow,square,sqrt,sum,Tensor,empty,add
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class CondInsNorm(nn.Module):
    ''' Implementing conditional instance normalization
        where input_x is normalized wrt input_z
        input_x is assumed to have the shape (N, x_dim, H, W)
        input_z is assumed to have the shape (N, z_dim, 1, 1) 
    '''
    def __init__(self,x_dim,z_dim,eps=1.0e-6,act_param=0.1):
        super(CondInsNorm,self).__init__()
        self.eps     = eps
        self.z_shift = nn.Sequential(
                             nn.Conv2d(in_channels  = z_dim,
                                       out_channels = x_dim,
                                       kernel_size  = 1,
                                       stride       = 1),
                             nn.ELU(alpha=act_param)
                             )
        self.z_scale = nn.Sequential(
                             nn.Conv2d(in_channels  = z_dim,
                                       out_channels = x_dim,
                                       kernel_size  = 1,
                                       stride       = 1),
                             nn.ELU(alpha=act_param)
                             )

    def forward(self,input_x,input_z):
        x_size  = input_x.size()
        assert len(x_size) == 4
        assert len(input_z.size()) == 4

        shift   = self.z_shift(input_z)
        scale   = self.z_scale(input_z)
        x_reshaped = input_x.view(x_size[0],x_size[1],x_size[2]*x_size[3]) 
        x_mean  = x_reshaped.mean(2,keepdim=True)
        x_var   = x_reshaped.var(2,keepdim=True)  
        x_rstd  = rsqrt(x_var + self.eps) # reciprocal sqrt
        x_s     = ((x_reshaped - x_mean)*x_rstd).view(*x_size) 
        output  = x_s*scale + shift
        return output


class InsNorm(nn.Module):
    ''' Implementing conditional instance normalization
        where input_x is normalized along the feature direction
        This is different from the BatchNorm base implementation
        in Pytorch
        input_x is assumed to have the shape (N, x_dim, H, W)
    '''
    def __init__(self,x_dim,eps=1.0e-6,act_param=0.1,affine=True):
        super(InsNorm,self).__init__()
        self.eps   = eps
        self.scale = nn.Parameter(Tensor(x_dim))
        self.shift = nn.Parameter(Tensor(x_dim))
        self.affine = affine
        # initialize weights
        if self.affine:
            nn.init.normal_(self.shift)
            nn.init.normal_(self.scale)

    def forward(self,input_x):
        x_size  = input_x.size()
        assert len(x_size) == 4
        x_reshaped = input_x.view(x_size[0],x_size[1],x_size[2]*x_size[3]) 
        x_mean  = x_reshaped.mean(2,keepdim=True)
        x_var   = x_reshaped.var(2,keepdim=True)  
        x_rstd  = rsqrt(x_var + self.eps) # reciprocal sqrt
        x_s     = ((x_reshaped - x_mean)*x_rstd).view(*x_size) 
        
        if self.affine:
            output  = x_s*self.scale[:,None,None] + self.shift[:,None,None]
        else:
            output  = x_s    
        return output        
    

class ApplyNormalization(nn.Module):
    ''' Normalizing input_x.
        input_x is assumed to have the shape (N, x_dim, H, W)
        If used, input_z is assumed to have the shape (N, z_dim, 1, 1) 
    '''
    def __init__(self,x_shape,z_dim=None,normalization=None):
        '''
        NOTE: x_shape does not include the number of samples N.
              Thus, x_shape[0] will give the channel size
        '''
        super(ApplyNormalization, self).__init__()
        if normalization =='cin':
            assert z_dim is not None
            self.xnorm = CondInsNorm(x_shape[0],z_dim)
        elif normalization == 'bn':
            self.xnorm = nn.BatchNorm2d(x_shape[0])
        elif normalization == 'ln':
            self.xnorm = nn.LayerNorm(x_shape)
        elif normalization == 'in':
            self.xnorm = InsNorm(x_shape[0],affine=True)    
        else:
            self.xnorm = nn.Identity()

    def forward(self,input_x,input_z=None):
        if input_z is None:
            out = self.xnorm(input_x)
        else:
            out = self.xnorm(input_x,input_z)      
        return out            

            


class ResBlock(nn.Module):
    ''' Implementing a single ResBlock
        input_x is assumed to have the shape (N, x_dim, H, W)
        If used, input_z is assumed to have the shape (N, z_dim, 1, 1) 
    '''
    def __init__(self,x_shape,z_dim=None,normalization=None,act_param=0.1):
        '''
        x_shape does not include the number of samples N.
        '''
        super(ResBlock,self).__init__()

        self.norm  = ApplyNormalization(x_shape,z_dim,normalization)
        self.conv  = nn.Conv2d(in_channels  = x_shape[0],
                               out_channels = x_shape[0],
                               kernel_size = 1,
                               stride      = 1)
        self.branch = nn.ModuleList(
                         self.build_branch(x_shape,
                                           z_dim,
                                           normalization,
                                           act_param)
                         )
        # for i, layer in enumerate(self.branch):
        #     self.add_module(f"branch_layer_{i}", layer)

    def build_branch(self,x_shape,z_dim,normalization,act_param):
        '''
        x_shape does not include the number of samples N.
        '''
        model = [nn.ELU(alpha=act_param),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(in_channels   = x_shape[0],
                           out_channels  = x_shape[0],
                           kernel_size   = 3,
                           stride        = 1),
                 ApplyNormalization(x_shape,z_dim,normalization),
                 nn.ELU(alpha=act_param),  
                 nn.ReflectionPad2d(1),  
                 nn.Conv2d(in_channels   = x_shape[0],
                           out_channels  = x_shape[0],
                           kernel_size  = 3,
                           stride       = 1)
                ]                   
        return model 

    def forward(self,input_x,input_z=None):
        x  = self.norm(input_x,input_z)  
        x1 = self.conv(x)     
        # Hacky way of using variable number of inputs per layer
        for i, layer in enumerate(self.branch):
            if i==3: # This is the normalization layer
              x = layer(x,input_z)
            else:
              x = layer(x)  
        output = x + x1
        return output

class DenseBlock(nn.Module):
    ''' Implementing a single DenseBlock (see Densely Connected Convolution Networks by Huang et al.)
        input_x is assumed to have the shape (N, x_dim, H, W)
        If used, input_z is assumed to have the shape (N, z_dim, 1, 1) 
    '''
    def __init__(self,x_shape,z_dim=None,normalization=None,act_param=0.1,out_channels=16,layers=4):
        '''
        x_shape does not include the number of samples N.
        '''
        super(DenseBlock,self).__init__()

        self.model  = nn.ModuleList(self.build_block(x_shape=x_shape,
                                                     z_dim=z_dim,
                                                     normalization=normalization,
                                                     act_param=act_param,
                                                     out_channels=out_channels,
                                                     layers=layers))

        self.layers=layers

    def build_block(self,x_shape,z_dim,normalization,act_param,out_channels,layers):
        '''
        x_shape does not include the number of samples N.
        '''
        model = []

        for i in range(layers):
            x_shape_i = (x_shape[0]+i*out_channels,x_shape[1],x_shape[2])
            model.append(ApplyNormalization(x_shape_i,z_dim,normalization))
            model.append(nn.ELU(alpha=act_param))
            model.append(nn.ReflectionPad2d(1))
            if i < layers-1:
                model.append(nn.Conv2d(in_channels   = x_shape_i[0],
                                       out_channels  = out_channels,
                                       kernel_size   = 3,
                                       stride        = 1))
            else: # Last layer has same number of output channels as initial input
                model.append(nn.Conv2d(in_channels   = x_shape_i[0],
                                       out_channels  = x_shape[0],
                                       kernel_size   = 3,
                                       stride        = 1))
                    
        return model 

    def forward(self,input_x,input_z=None):  
        # Hacky way of using variable number of inputs per layer
        for i in range(0,4*self.layers,4):
            if i==0:
                x_in = input_x
            else:
                x_in = cat((x_in,x),dim=1)   
            x = self.model[i](x_in,input_z)
            x = self.model[i+1](x)
            x = self.model[i+2](x)
            x = self.model[i+3](x)
 
        return x       

class DownSample(nn.Module):
    ''' Implementing a downsampling using average pooling
        input_x is assumed to have the shape (N, x_dim, H, W)
    '''
    def __init__(self,x_dim,filters,downsample=True,activation=True,act_param=0.1,ds_k=2,ds_s=2):
        super(DownSample, self).__init__()
        self.pad  = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels   = x_dim,
                              out_channels  = filters,
                              kernel_size  = 3,
                              stride       = 1)
        self.LReLU = nn.ELU(alpha=act_param)
        self.pool  = nn.AvgPool2d(kernel_size=ds_k,stride=ds_s)
        self.downsample = downsample
        self.activation = activation

    def forward(self,input_x):
        x = self.pad(input_x)
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.downsample:
            x = self.pool(x)
        return x      

class UpSample(nn.Module):
    ''' Implementing a upsampling with skip connection
        concatenations 
        input_x is assumed to have the shape (N, x_dim, H, W)
        If used, old_x is assumed to have size (N, old_x_dim, H, W)
        where old_x_dim need not be the same as x_dim
    '''
    def __init__(self,x_dim,filters,upsample=True, concat=False,
                 old_x_dim=0,activation=True,act_param=0.1):
        super(UpSample, self).__init__()
        self.upsample   = upsample
        self.activation = activation
        self.concat     = concat

        if self.concat:
            input_dim = x_dim + old_x_dim
        else:
            input_dim = x_dim     
        self.pad  = nn.ReflectionPad2d(1)      
        self.conv = nn.Conv2d(in_channels   = input_dim,
                              out_channels  = filters,
                              kernel_size   = 3,
                              stride        = 1)
        self.LReLU = nn.ELU(alpha=act_param)
        self.rpool = nn.Upsample(scale_factor=2)
        

    def forward(self,input_x,old_x=None):
        if self.concat and old_x is not None:
            x = cat((input_x,old_x),dim=1)
        else:
            x = input_x  
        x = self.pad(x)      
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.upsample:
            x = self.rpool(x)
        return x           


class generator_resskip(nn.Module):
    ''' Generator model: U-Net with skip connections and Resblocks
        input_x is assumed to have the shape (N, C, H, W)
        input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    '''
    def __init__(self,x_shape,z_dim=None,k0=32,act_param=0.1,g_out='x'):
        '''
        x_shape does not include the number of samples N.
        '''
        super(generator_resskip, self).__init__()
        C0,H0,W0 = x_shape

        if z_dim == None:
            normalization = 'in'
        else:
            normalization = 'cin'    

        # ------ Down branch -----------------------------
        H,W,k=H0,W0,k0
        self.d1 = DownSample(x_dim=C0,
                             filters=k,
                             downsample=False,
                             act_param=act_param)
        self.d2 = ResBlock(x_shape=(k,H,W),
                           act_param=act_param)
        
        self.d3 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k 
        self.d4 = ResBlock(x_shape=(k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.d5 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k  
        self.d6 = ResBlock(x_shape=(k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.d7 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k  
        self.d8 = ResBlock(x_shape=(k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)
        # ------------------------------------------------

        # ----- Base of UNet------------------------------ 
        self.base = ResBlock(x_shape=(k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param)
        #-------------------------------------------------

        # ------ Up branch -----------------------------
        self.u1 = UpSample(x_dim=k,filters=k,
                           act_param=act_param)
        H,W,k = 2*H,2*W,k//2
        self.u2 = ResBlock(x_shape=(2*k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.u3 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim = k,
                           act_param=act_param)
        H,W,k = 2*H,2*W,k//2
        self.u4 = ResBlock(x_shape=(2*k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.u5 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim =k,
                           act_param=act_param)

        H,W,k = 2*H,2*W,k//2
        self.u6 = ResBlock(x_shape=(2*k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.u7 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim =k,
                           act_param=act_param,
                           upsample=False)
        
        self.u8 = ResBlock(x_shape=(k,H,W),
                           z_dim=z_dim,
                           normalization=normalization,
                           act_param=act_param)

        self.u9 = UpSample(x_dim=k,
                           filters=C0,
                           upsample=False,
                           activation=False)


        if g_out == 'x':
            self.res_wt = 0.0
            self.ofunc = nn.Sigmoid()
        elif g_out == 'dx':
            self.res_wt = 1.0
            self.ofunc = nn.Identity()
        elif g_out == 'bdx':
            self.res_wt = 1.0
            self.ofunc = nn.Sigmoid()    
        # ------------------------------------------------

    def forward(self,input_x,input_z=None):
        
        x1 = self.d1(input_x=input_x)
        x1 = self.d2(input_x=x1)

        x2 = self.d3(input_x=x1)
        x2 = self.d4(input_x=x2,input_z=input_z)

        x3 = self.d5(input_x=x2)
        x3 = self.d6(input_x=x3,input_z=input_z)

        x4 = self.d7(input_x=x3)
        x4 = self.d8(input_x=x4,input_z=input_z)

        x5 = self.base(input_x=x4,input_z=input_z)

        x6 = self.u1(input_x=x5)
        x6 = self.u2(input_x=x6,input_z=input_z)

        x7 = self.u3(input_x=x6,old_x=x3)
        x7 = self.u4(input_x=x7,input_z=input_z)

        x8 = self.u5(input_x=x7,old_x=x2)
        x8 = self.u6(input_x=x8,input_z=input_z)

        x9 = self.u7(input_x=x8,old_x=x1)
        x9 = self.u8(input_x=x9,input_z=input_z)

        x10 = self.u9(input_x=x9)

        output = self.ofunc(x10 + self.res_wt*input_x)
        #output = x10 + self.res_wt*input_x
        #output = self.ofunc(x10 + self.res_wt*input_x) - self.ofunc(1.0-x10 - self.res_wt*input_x)

        return output      

class generator_denseskip(nn.Module):
    ''' Generator model: U-Net with skip connections and Denseblocks
        input_x is assumed to have the shape (N, C, H, W)
        input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    '''
    def __init__(self,x_shape,z_dim=50,k0=32,act_param=0.1,denselayers=4,dense_int_out=16,g_out='x'):
        '''
        x_shape does not include the number of samples N.
        '''
        super(generator_denseskip, self).__init__()
        C0,H0,W0 = x_shape

        if z_dim == None:
            normalization = 'in'
        else:
            normalization = 'cin'

        # ------ Down branch -----------------------------
        H,W,k=H0,W0,k0
        self.d1 = DownSample(x_dim=C0,
                             filters=k,
                             downsample=False,
                             act_param=act_param)
        self.d2 = DenseBlock(x_shape=(k,H,W),
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)
        
        self.d3 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k 
        self.d4 = DenseBlock(x_shape=(k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.d5 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k  
        self.d6 = DenseBlock(x_shape=(k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.d7 = DownSample(x_dim=k,filters=2*k,
                             act_param=act_param)
        H,W,k = (H-2)//2 + 1,(W-2)//2 + 1,2*k  
        self.d8 = DenseBlock(x_shape=(k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)
        # ------------------------------------------------

        # ----- Base of UNet------------------------------ 
        self.base = DenseBlock(x_shape=(k,H,W),
                               z_dim=z_dim,
                               normalization=normalization,
                               act_param=act_param,
                               out_channels=dense_int_out,
                               layers=denselayers)
        #-------------------------------------------------

        # ------ Up branch -----------------------------
        self.u1 = UpSample(x_dim=k,filters=k,
                           act_param=act_param)
        H,W,k = 2*H,2*W,k//2
        self.u2 = DenseBlock(x_shape=(2*k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.u3 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim = k,
                           act_param=act_param)
        H,W,k = 2*H,2*W,k//2
        self.u4 = DenseBlock(x_shape=(2*k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.u5 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim =k,
                           act_param=act_param)

        H,W,k = 2*H,2*W,k//2
        self.u6 = DenseBlock(x_shape=(2*k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.u7 = UpSample(x_dim=2*k,
                           filters=k,
                           concat=True,
                           old_x_dim =k,
                           act_param=act_param,
                           upsample=False)
        
        self.u8 = DenseBlock(x_shape=(k,H,W),
                             z_dim=z_dim,
                             normalization=normalization,
                             act_param=act_param,
                             out_channels=dense_int_out,
                             layers=denselayers)

        self.u9 = UpSample(x_dim=k,
                           filters=C0,
                           upsample=False,
                           activation=False)

        if g_out == 'x':
            self.res_wt = 0.0
            self.ofunc = nn.Sigmoid()
        elif g_out == 'dx':
            self.res_wt = 1.0
            self.ofunc = nn.Identity()
        elif g_out == 'bdx':
            self.res_wt = 1.0
            self.ofunc = nn.Sigmoid()
        # ------------------------------------------------

    def forward(self,input_x,input_z=None):
        
        x1 = self.d1(input_x=input_x)
        x1 = self.d2(input_x=x1)

        x2 = self.d3(input_x=x1)
        x2 = self.d4(input_x=x2,input_z=input_z)

        x3 = self.d5(input_x=x2)
        x3 = self.d6(input_x=x3,input_z=input_z)

        x4 = self.d7(input_x=x3)
        x4 = self.d8(input_x=x4,input_z=input_z)

        x5 = self.base(input_x=x4,input_z=input_z)

        x6 = self.u1(input_x=x5)
        x6 = self.u2(input_x=x6,input_z=input_z)

        x7 = self.u3(input_x=x6,old_x=x3)
        x7 = self.u4(input_x=x7,input_z=input_z)

        x8 = self.u5(input_x=x7,old_x=x2)
        x8 = self.u6(input_x=x8,input_z=input_z)

        x9 = self.u7(input_x=x8,old_x=x1)
        x9 = self.u8(input_x=x9,input_z=input_z)

        x10 = self.u9(input_x=x9)

        output = self.ofunc(x10 + self.res_wt*input_x)
        #output = x10 + self.res_wt*input_x 
        #output = self.ofunc(x10 + self.res_wt*input_x) - self.ofunc(1.0-x10 - self.res_wt*input_x)

        return output     

class critic_res(nn.Module):
    ''' Critic model using Resblocks
        input_x and input_y are both assumed to have 
        the shape (N, C, H, W)
    '''
    def __init__(self,x_shape,k0=32,act_param=0.1):
        '''
        x_shape does not include the number of samples N.
        '''
        super(critic_res, self).__init__()
        C0,H0,W0 = x_shape

        # ------ Convolution layers -----------------------------
        H,W=H0,W0
        self.cnn1 = DownSample(x_dim=2*C0,
                               filters=k0,
                               downsample=False,
                               act_param=act_param)
        self.cnn2 = ResBlock(x_shape=(k0,H,W),
                            act_param=act_param)
        
        self.cnn3 = DownSample(x_dim=k0,filters=2*k0,
                             act_param=act_param,
                             ds_k=4,ds_s=4)
        H,W = (H-2)//4 + 1,(W-2)//4 + 1 
        self.cnn4 = ResBlock(x_shape=(2*k0,H,W),
                           normalization='ln',
                           act_param=act_param)

        self.cnn5 = DownSample(x_dim=2*k0,filters=4*k0,
                             act_param=act_param,
                             ds_k=4,ds_s=4)
        H,W = (H-2)//4 + 1,(W-2)//4 + 1 
        self.cnn6 = ResBlock(x_shape=(4*k0,H,W),
                           normalization='ln',
                           act_param=act_param)

        # self.cnn7 = DownSample(x_dim=4*k0,filters=8*k0,
        #                      act_param=act_param,
        #                      ds_k=4,ds_s=4)
        # H,W = (H-2)//2 + 1,(W-2)//2 + 1 
        # self.cnn8 = ResBlock(x_shape=(8*k0,H,W),
        #                    normalization='in',
        #                    act_param=act_param)

        # ------------------------------------------------

        # ----- Dense layers------------------------------ 
        self.flat  = nn.Flatten()
        self.lin1  = nn.Linear(in_features=4*k0*H*W,
                               out_features=128)
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN    = ApplyNormalization(x_shape=(128),normalization='ln')
        self.lin2  = nn.Linear(in_features=128,
                               out_features=1)
  
        # ------------------------------------------------

    def forward(self,input_x,input_y):
        
        xy = cat((input_x,input_y),dim=1)

        x = self.cnn1(input_x=xy)
        x = self.cnn2(input_x=x)
        x = self.cnn3(input_x=x)
        x = self.cnn4(input_x=x)
        x = self.cnn5(input_x=x)
        x = self.cnn6(input_x=x)
        # x = self.cnn7(input_x=x)
        # x = self.cnn8(input_x=x)

        x = self.flat(x)
        x = self.lin1(x)
        x = self.LReLU(x)
        x = self.LN(x)
        output = self.lin2(x)

        return output

class critic_dense(nn.Module):
    ''' Critic model using Denseblocks
        input_x and input_y are both assumed to have 
        the shape (N, C, H, W)
    '''
    def __init__(self,x_shape,k0=20,act_param=0.1,denselayers=4,dense_int_out=16):
        '''
        x_shape does not include the number of samples N.
        '''
        super(critic_dense, self).__init__()
        C0,H0,W0 = x_shape

        # ------ Convolution layers -----------------------------
        H,W=H0,W0
        self.cnn1 = DownSample(x_dim=2*C0,
                               filters=k0,
                               downsample=False,
                               act_param=act_param)

        self.cnn2 = DenseBlock(x_shape=(k0,H,W),
                               act_param=act_param,
                               out_channels=dense_int_out,
                               layers=denselayers)
        
        self.cnn3 = DownSample(x_dim=k0,filters=2*k0,
                             act_param=act_param,
                             ds_k=4,ds_s=4)
        H,W = (H-2)//4 + 1,(W-2)//4 + 1 
        self.cnn4 = DenseBlock(x_shape=(2*k0,H,W),
                               act_param=act_param,
                               normalization='ln',
                               out_channels=dense_int_out,
                               layers=denselayers)

        self.cnn5 = DownSample(x_dim=2*k0,filters=4*k0,
                             act_param=act_param,
                             ds_k=4,ds_s=4)
        H,W = (H-2)//4 + 1,(W-2)//4 + 1 

        self.cnn6 = DenseBlock(x_shape=(4*k0,H,W),
                               act_param=act_param,
                               normalization='ln',
                               out_channels=dense_int_out,
                               layers=denselayers)

        self.cnn7 = DownSample(x_dim=4*k0,filters=8*k0,
                             act_param=act_param,
                             ds_k=4,ds_s=4)
        H,W = (H-2)//4 + 1,(W-2)//4 + 1 
        self.cnn8 = DenseBlock(x_shape=(8*k0,H,W),
                               act_param=act_param,
                               normalization='ln',
                               out_channels=dense_int_out,
                               layers=denselayers)
        # ------------------------------------------------

        # ----- Dense layers------------------------------ 
        self.flat  = nn.Flatten()
        self.lin1  = nn.Linear(in_features=8*k0*H*W,
                               out_features=128)
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN    = ApplyNormalization(x_shape=(128),normalization='ln')
        self.lin2  = nn.Linear(in_features=128,
                               out_features=1)
  
        # ------------------------------------------------

    def forward(self,input_x,input_y):
        
        xy = cat((input_x,input_y),dim=1)

        x = self.cnn1(input_x=xy)
        x = self.cnn2(input_x=x)
        x = self.cnn3(input_x=x)
        x = self.cnn4(input_x=x)
        x = self.cnn5(input_x=x)
        x = self.cnn6(input_x=x)
        x = self.cnn7(input_x=x)
        x = self.cnn8(input_x=x)

        x = self.flat(x)
        x = self.lin1(x)
        x = self.LReLU(x)
        x = self.LN(x)
        output = self.lin2(x)

        return output


def get_lat_var(batch_size,x_W,x_H,z_dim):
    ''' This function generates latent variables'''
    if z_dim == None:
        z = None
    else: 
        z = randn((batch_size,z_dim,1,1))  
    return z
    
def gradient_penalty(fake_X, true_X, true_Y,model, device, p=2, c0=1.0):
    '''Evaluates gradient penalty term'''
    batch_size, *other_dims = true_X.size()
    epsilon = rand([batch_size]+[1 for _ in range(len(other_dims))])
    epsilon = epsilon.expand(-1,*other_dims).to(device)
    x_hat   = epsilon * true_X + (1 - epsilon) * fake_X
    x_hat.requires_grad=True
    d_hat   = model(x_hat,true_Y)
    grad    = autograd.grad(outputs=d_hat,
                          inputs=x_hat,
                          grad_outputs=ones_like(d_hat).to(device),
                          create_graph=True,
                          retain_graph=True)[0]
    grad = grad.view(batch_size,-1)
    grad_norm  = sqrt(1.0e-8 + sum(square(grad),dim=1))
    grad_penalty = pow(grad_norm-c0, p).mean()
    return grad_penalty

def full_gradient_penalty(fake_X, true_X, true_Y,model, device, p=2, c0=1.0):
    '''Evaluates full gradient penalty term'''
    batch_size, *other_dims = true_X.size()
    epsilon  = rand([batch_size]+[1 for _ in range(len(other_dims))])
    epsilon  = epsilon.expand(-1,*other_dims).to(device)
    x_hat    = epsilon * true_X + (1 - epsilon) * fake_X
    x_hat.requires_grad=True
    true_Y.requires_grad=True
    d_hat    = model(x_hat,true_Y)
    grad     = autograd.grad(outputs=d_hat,
                            inputs=(x_hat, true_Y),
                            grad_outputs=ones_like(d_hat).to(device),
                            create_graph=True,
                            retain_graph=True)
    grad_x, grad_y = grad[0], grad[1]
    grad_x = grad_x.view(batch_size,-1)
    grad_y = grad_y.view(batch_size,-1)
    grad_norm  = sqrt(1.0e-8 + add(sum(square(grad_x),dim=1),sum(square(grad_y),dim=1)))
    grad_penalty = pow(grad_norm-c0, p).mean()
    return grad_penalty
 
