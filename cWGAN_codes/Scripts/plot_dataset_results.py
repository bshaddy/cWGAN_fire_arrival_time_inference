# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from skimage.color import rgb2yuv, yuv2rgb
from config import cla
from utils import *
from data_utils import *

def plot_results(PARAMS):

    if PARAMS.GANdir == None:
        GANdir  = get_GAN_dir(PARAMS)
    else:
        GANdir  = PARAMS.GANdir 
    results_dir = f"{GANdir}/{PARAMS.results_dir}"
    ckpt        = PARAMS.ckpt_id

    if PARAMS.z_dim == None:
        z_n_MC = 1
        nb_    = 3
        qoi    = 'mode'
    else:    
        z_n_MC = PARAMS.z_n_MC
        nb_    = 4
        qoi    = 'mean'

    cmap       = 'viridis'
    fontsize   = 30
    labelsize  = 15

    file_list = glob.glob(f'{results_dir}/stat*_{ckpt}.npy')
    file_list.sort()

    n_samples = len(file_list)

    psnr_ = np.zeros(n_samples)
    nmse_ = np.zeros(n_samples)
    ssim_ = np.zeros(n_samples)
    sd_   = np.zeros(n_samples)

    for i in range(n_samples):

        data = np.load(file_list[i])
        nimp = data.shape[0]-nb_ 

        nrow,ncol = 2,nb_
        fig = plt.figure(figsize=(5*nb_,5*nrow))
        grid = AxesGrid(fig,111,
                    nrows_ncols=(nrow, nb_),
                    axes_pad=0.50,
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="1%"
                     )
        ax_ind = 0

        axs = grid[ax_ind]
        cmin=0
        cmax=1
        cmid=0.5
        meas = data[1]
        meas[meas = 0] = np.nan
        meas[meas = 1] = 10
        masked_array = np.ma.array(meas, mask=np.isnan(meas))
        cmap.set_bad(alpha=None) 
        cmap.set_over(color='darkkhaki')
        pcm = axs.imshow(masked_array,aspect='equal',cmap=cmap,vmin=0,vmax=1)
        plt.colorbar(pcm,cax=grid.cbar_axes[ax_ind],ticks=[cmin,cmid,cmax],format='%.1f').ax.tick_params(labelsize=labelsize)
        #grid.cbar_axes[ax_ind].remove()
        axs.set_title(f'measurement',fontsize=fontsize)
        axs.axes.xaxis.set_visible(False)
        axs.axes.yaxis.set_visible(False)
        ax_ind +=1

        axs = grid[ax_ind]
        cmin = np.min(data[0])
        cmax = np.max(data[0])
        cmid = 0.5*(cmin+cmax)
        pcm = axs.imshow(data[0],aspect='equal',cmap=cmap)
        plt.colorbar(pcm,cax=grid.cbar_axes[ax_ind],ticks=[cmin,cmid,cmax],format='%.1f').ax.tick_params(labelsize=labelsize)
        # grid.cbar_axes[ax_ind].remove()
        axs.set_title(f'ground truth',fontsize=fontsize)
        axs.axes.xaxis.set_visible(False)
        axs.axes.yaxis.set_visible(False)
        ax_ind +=1

        
        # xmin = min(np.min(data[2]),0.0)
        # xmax = max(np.max(data[2]),1.0)
        cmin = np.min(data[2])
        cmax = np.max(data[2])
        cmid = 0.5*(cmin+cmax)
        axs = grid[ax_ind]
        pcm = axs.imshow(data[2],aspect='equal',cmap=cmap)
        plt.colorbar(pcm,cax=grid.cbar_axes[ax_ind],ticks=[cmin,cmid,cmax],format='%.1f').ax.tick_params(labelsize=labelsize)
        # grid.cbar_axes[ax_ind].remove()
        axs.set_title(f'{qoi}',fontsize=fontsize)
        axs.axes.xaxis.set_visible(False)
        axs.axes.yaxis.set_visible(False)
        ax_ind +=1

        nmse_[i],ssim_[i],psnr_[i] = eval_metric(true=data[0].squeeze(),gen=data[2].squeeze())

        if PARAMS.z_dim is not None:
            axs = grid[ax_ind]
            cmin = np.min(data[3])
            cmax = np.max(data[3])
            cmid = 0.5*(cmin+cmax)
            pcm = axs.imshow(data[3],aspect='equal',cmap=cmap)
            plt.colorbar(pcm,cax=grid.cbar_axes[ax_ind],ticks=[cmin,cmid,cmax],format='%.1f').ax.tick_params(labelsize=labelsize)
            #grid.cbar_axes[ax_ind].colorbar(pcm,ticks=[cmin,cmid,cmax],format='%.1f')
            #grid.cbar_axes[ax_ind].tick_params(labelsize=labelsize)
            axs.set_title(f'SD',fontsize=fontsize)
            axs.axes.xaxis.set_visible(False)
            axs.axes.yaxis.set_visible(False)
            ax_ind +=1

            sd_[i] = np.linalg.norm(data[3])/np.size(data[3])

        
        if nimp > 0:
            for j in range(nimp):
                axs = grid[ax_ind]
                cmin = np.min(data[4+j])
                cmax = np.max(data[4+j])
                cmid = 0.5*(cmin+cmax)
                pcm = axs.imshow(data[4+j],aspect='equal',cmap=cmap)
                plt.colorbar(pcm,cax=grid.cbar_axes[ax_ind],ticks=[cmin,cmid,cmax],format='%.1f').ax.tick_params(labelsize=labelsize)
                #grid.cbar_axes[ax_ind].remove()
                axs.set_title(f'imp. sample {j+1}',fontsize=fontsize)
                axs.axes.xaxis.set_visible(False)
                axs.axes.yaxis.set_visible(False)
                ax_ind +=1

        fname = file_list[i][0:-4]+'.pdf'
        fig.savefig(fname, bbox_inches = 'tight')
        plt.close('all')  


    # Saving and plotting metrics
    plt.figure()
    plt.plot(np.arange(1,n_samples+1),nmse_,'--o')
    plt.plot(np.arange(1,n_samples+1),np.mean(nmse_)*np.ones(n_samples),'-')
    plt.xlabel('Sample')
    plt.ylabel('NMSE')
    plt.grid()
    plt.savefig(f'{results_dir}/nmse_ckpt_{ckpt}.pdf', bbox_inches = 'tight')
    plt.close('all')  
    np.save(f'{results_dir}/nmse_ckpt_{ckpt}.npy',nmse_)

    plt.figure()
    plt.plot(np.arange(1,n_samples+1),psnr_,'--o')
    plt.plot(np.arange(1,n_samples+1),np.mean(psnr_)*np.ones(n_samples),'-')
    plt.xlabel('Sample')
    plt.ylabel('PSNR')
    plt.grid()
    plt.savefig(f'{results_dir}/psnr_ckpt_{ckpt}.pdf', bbox_inches = 'tight')
    plt.close('all')  
    np.save(f'{results_dir}/psnr_ckpt_{ckpt}.npy',psnr_)


    plt.figure()
    plt.plot(np.arange(1,n_samples+1),ssim_,'--o')
    plt.plot(np.arange(1,n_samples+1),np.mean(ssim_)*np.ones(n_samples),'-')
    plt.xlabel('Sample')
    plt.ylabel('SSIM')
    plt.grid()
    plt.savefig(f'{results_dir}/ssim_ckpt_{ckpt}.pdf', bbox_inches = 'tight')
    plt.close('all')  
    np.save(f'{results_dir}/ssim_ckpt_{ckpt}.npy',ssim_)

    if PARAMS.z_dim is not None:
        plt.figure()
        plt.plot(np.arange(1,n_samples+1),sd_,'--o')
        plt.plot(np.arange(1,n_samples+1),np.mean(sd_)*np.ones(n_samples),'-')
        plt.xlabel('Sample')
        plt.ylabel('SD')
        plt.grid()
        plt.savefig(f'{results_dir}/sd_ckpt_{ckpt}.pdf', bbox_inches = 'tight')
        plt.close('all')  

        plt.figure()
        plt.plot(sd_,ssim_,'o')
        plt.xlabel('SD')
        plt.ylabel('SSIM')
        plt.grid()
        plt.savefig(f'{results_dir}/sd_vs_ssim_ckpt_{ckpt}.pdf', bbox_inches = 'tight')
        plt.close('all')

        np.save(f'{results_dir}/sd_ckpt_{ckpt}.npy',sd_)
    

if __name__=="__main__":
    print("Generating plots")
    PARAMS = cla()
    plot_results(PARAMS)
    print("----------------------- DONE --------------------------")



 
