# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os,shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2yuv, yuv2rgb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def make_save_dir(PARAMS):
    ''' This function creates the results save directory'''

    savedir = get_GAN_dir(PARAMS)

    if os.path.exists(savedir):
        print('\n     *** Folder already exists!\n')    
    else:
        os.makedirs(savedir)

    # Creating directory to save generator checkpoints
    if os.path.exists(f"{savedir}/checkpoints"):
        print('\n     *** Checkpoints directory already exists\n')    
    else:
        os.makedirs(f"{savedir}/checkpoints")    

    print('\n --- Saving parameters to file \n')
    param_file = savedir + '/parameters.txt'
    with open(param_file,"w") as fid:
        for pname in vars(PARAMS):
            fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")    

    return savedir 
 
def get_GAN_dir(PARAMS): 

    if PARAMS.z_dim == None:
        z_dim = 0
    else:
        z_dim = PARAMS.z_dim    
    savedir = f"../exps/g_{PARAMS.g_type}"\
              f"_{PARAMS.g_out}"\
              f"_d_{PARAMS.d_type}"\
              f"_Nsamples{PARAMS.n_train}"\
              f"_Ncritic{PARAMS.n_critic}_Zdim{z_dim}"\
              f"_BS{PARAMS.batch_size}_Nepoch{PARAMS.n_epoch}"\
              f"_act_{PARAMS.act_param}"\
              f"_GP{PARAMS.gp_coef}{PARAMS.sdir_suffix}"      
    return savedir    

def save_loss(loss,loss_name,savedir,n_epoch,ckpt_id):

    with open(f"{savedir}/{loss_name}.txt","ab") as f:
        np.savetxt(f,loss)    
    fig, ax1 = plt.subplots()
    loss_restart = np.loadtxt(f"{savedir}/{loss_name}.txt")    
    ax1.plot(loss_restart)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss_restart)])


    ax2 = ax1.twiny()
    xmax = ckpt_id + n_epoch
    ax2.set_xlim([0,xmax])
    ax2.set_xlabel('Epochs')


    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)    
    plt.close()           
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_metric(true,gen):
    '''
    Calculate PSNR, SSIM and MSE for Y channels of images
    '''    
    assert true.shape == gen.shape
    assert len(true.shape) == 2
    nmse_ = np.linalg.norm(true-gen)/np.linalg.norm(true)
    ssim_ = ssim(true,gen)
    psnr_ = psnr(true,gen)
    return nmse_,ssim_,psnr_       


    
