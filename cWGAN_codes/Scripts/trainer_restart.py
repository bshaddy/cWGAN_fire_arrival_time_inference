# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from torchinfo import summary
from functools import partial
import time,random
from config import cla
from utils_restart import *
from data_utils import *
from models import *
import timeit



if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)


PARAMS = cla()

# assert PARAMS.z_dim == None or PARAMS.z_dim > 0

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)

print('\n ============== LAUNCHING TRAINING SCRIPT =================\n')


print('\n --- Creating network folder \n')
savedir  = PARAMS.GANdir


print('\n --- Loading training data from file\n')
train_data   = LoadDataset(datafile   = PARAMS.train_file,
                           N          = PARAMS.n_train,
                           permute    = True)  

_,x_C,x_W,x_H = train_data.x_shape

# Creating sub-function to generate latent variables
glv    = partial(get_lat_var,x_W=x_W,x_H=x_H,z_dim=PARAMS.z_dim)
    
print('\n --- Loading conditional GAN models from checkpoint\n')

if PARAMS.GANdir == None:
    GANdir = get_GAN_dir(PARAMS)
else:
    GANdir = PARAMS.GANdir

if PARAMS.g_type == 'resskip':
    G_model = generator_resskip(x_shape=(x_C,x_H,x_W),z_dim=PARAMS.z_dim,g_out=PARAMS.g_out,act_param=PARAMS.act_param)  
elif PARAMS.g_type == 'denseskip':
    G_model = generator_denseskip(x_shape=(x_C,x_H,x_W),z_dim=PARAMS.z_dim,denselayers=3,g_out=PARAMS.g_out,act_param=PARAMS.act_param)    

if PARAMS.d_type == 'res':    
    D_model = critic_res(x_shape=(x_C,x_H,x_W),act_param=PARAMS.act_param)
elif PARAMS.d_type == 'dense':    
    D_model = critic_dense (x_shape=(x_C,x_H,x_W),denselayers=2,act_param=PARAMS.act_param)

if PARAMS.z_dim == None:
    summary(G_model,input_size=(2,x_C,x_H,x_W))
    z_n_MC = 1
    ncol   = 3
    qoi    = 'mode'
else:
    summary(G_model,input_size=[(2,x_C,x_H,x_W),(2,PARAMS.z_dim,1,1)])      
    z_n_MC = PARAMS.z_n_MC
    ncol = 4
    qoi    = 'mean'
summary(D_model,input_size=[(2,x_C,x_H,x_W),(2,x_C,x_H,x_W)])   

# Moving models to correct device and adding optimizers
G_model.to(device)
D_model.to(device)
G_optim = torch.optim.Adam(G_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param)
D_optim = torch.optim.Adam(D_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param)

# Load model checkpoint
G_state = torch.load(f"{GANdir}/checkpoints/G_model_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
G_model.load_state_dict(G_state)
D_state = torch.load(f"{GANdir}/checkpoints/D_model_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
D_model.load_state_dict(D_state)
G_optim_state = torch.load(f"{GANdir}/checkpoints/G_optim_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
G_optim.load_state_dict(G_optim_state)
D_optim_state = torch.load(f"{GANdir}/checkpoints/D_optim_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
D_optim.load_state_dict(D_optim_state)

# Creating data loader and any other dummmy/display variables
loader = DataLoader(train_data, batch_size=PARAMS.batch_size, shuffle=True, drop_last=True)
n_view = 5 # Number of training samples to save plots
stat_z = glv(batch_size=z_n_MC*n_view)
view_x,view_y = train_data[0:n_view]
stat_y = torch.repeat_interleave(view_y,repeats=z_n_MC,dim=0)
stat_y = stat_y.to(device)
if PARAMS.z_dim is not None:
    stat_z = stat_z.to(device)


# ============ Training ==================
print('\n --- Initiating GAN training\n')

n_iters = 1
G_loss_log    = []
D_loss_log    = []
mismatch_log  = []
wd_loss_log   = [] 

start_time = timeit.default_timer()

for i in range(PARAMS.n_epoch):
    for true_X,true_Y in loader:

        true_X = true_X.to(device)
        true_Y = true_Y.to(device)

        # ---------------- Updating critic -----------------------
        D_optim.zero_grad()
        z = glv(batch_size=PARAMS.batch_size)
        if PARAMS.z_dim is not None:
            z = z.to(device)
        fake_X     = G_model(true_Y,z).detach()
        fake_val   = D_model(fake_X,true_Y)
        true_val   = D_model(true_X,true_Y)
        gp_val     = full_gradient_penalty(fake_X=fake_X, 
                                      true_X=true_X, 
                                      true_Y=true_Y,
                                      model =D_model, 
                                      device=device)
        fake_loss  = torch.mean(fake_val)
        true_loss  = torch.mean(true_val)
        wd_loss    = true_loss - fake_loss
        D_loss     = -wd_loss + PARAMS.gp_coef*gp_val

        D_loss.backward()
        D_optim.step()
        D_loss_log.append(D_loss.item())
        wd_loss_log.append(wd_loss.item())
        epoch_tracker = PARAMS.ckpt_id + i
        print(f"     *** (epoch total,iter since restart):({epoch_tracker},{n_iters}) ---> d_loss:{D_loss.item():.4e}, gp_term:{gp_val.item():.4e}, wd:{wd_loss.item():.4e}")

        # ---------------- Updating generator -----------------------
        if n_iters%PARAMS.n_critic == 0:
            G_optim.zero_grad()
            z = glv(batch_size=PARAMS.batch_size)
            if PARAMS.z_dim is not None:
                z = z.to(device)
            fake_X     = G_model(true_Y,z)
            fake_val   = D_model(fake_X,true_Y)
            G_loss     = -torch.mean(fake_val)
            mismatch   = torch.mean(torch.linalg.matrix_norm(true_X-fake_X,ord=2))
            #mismatch   = torch.mean(torch.linalg.norm(true_X-fake_X,dim=(1,2,3)))/(x_C*x_W*x_H)
            #G_loss += PARAMS.mismatch_param*mismatch
            G_loss.backward()
            G_optim.step()
            G_loss_log.append(G_loss.item())
            mismatch_log.append(mismatch.item())
            print(f"     ***           ---> g_loss:{G_loss.item():.4e}, mismatch:{mismatch.item():.4e}") 
            elapsed = timeit.default_timer() - start_time
            print(elapsed)
            start_time = timeit.default_timer()

        
        n_iters += 1    

    # Saving intermediate output and checkpoints
    if (i+1) % PARAMS.save_freq == 0: 
        with torch.no_grad():
            print(f"     *** generating sample plots and saving generator checkpoint")
            G_model.eval()  # NOTE: switching to eval mode for generator  

            pred_ = G_model(stat_y,stat_z).cpu().detach().numpy().squeeze()
            G_model.train() # NOTE: switching back to train mode 

        fig1,axs1 = plt.subplots(n_view, ncol, dpi=100, figsize=(ncol*5,n_view*5))
        ax1 = axs1.flatten()
        ax_ind = 0
        for t in range(n_view):
            axs = ax1[ax_ind]
            imgy = view_y[t:t+1].numpy().squeeze()
            pcm  = axs.imshow(imgy,aspect='equal',cmap='viridis')
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'measurement',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            axs = ax1[ax_ind]
            imgx = view_x[t:t+1].numpy().squeeze()
            pcm = axs.imshow(imgx,aspect='equal',cmap='viridis')
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'target',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            sample_mean = np.mean(pred_[t*z_n_MC:(t+1)*z_n_MC],axis=0)
            
            axs = ax1[ax_ind]
            xmin = min(np.min(sample_mean),0.0)
            xmax = max(np.max(sample_mean),1.0)
            pcm = axs.imshow(sample_mean,aspect='equal',cmap='viridis')
            fig1.colorbar(pcm,ax=axs)
            if t==0:
                axs.set_title(f'{qoi}',fontsize=30)
            axs.axis('off')
            ax_ind +=1

            if ncol == 4:

                sample_std  = np.std(pred_[t*z_n_MC:(t+1)*z_n_MC],axis=0)
                axs = ax1[ax_ind]
                pcm = axs.imshow(sample_std,aspect='equal',cmap='viridis')
                fig1.colorbar(pcm,ax=axs)
                if t==0:
                    axs.set_title(f'std',fontsize=30)
                axs.axis('off')
                ax_ind +=1

        epoch_value = i+1+PARAMS.ckpt_id
        fig1.tight_layout()
        fig1.savefig(f"{savedir}/stats_val_sample_{epoch_value}.png")
        plt.close('all')

        torch.save(G_model.state_dict(),f"{savedir}/checkpoints/G_model_{epoch_value}.pth")
        torch.save(D_model.state_dict(),f"{savedir}/checkpoints/D_model_{epoch_value}.pth")
        torch.save(G_optim.state_dict(),f"{savedir}/checkpoints/G_optim_{epoch_value}.pth")
        torch.save(D_optim.state_dict(),f"{savedir}/checkpoints/D_optim_{epoch_value}.pth")

save_loss(G_loss_log,'g_loss',savedir,PARAMS.n_epoch,PARAMS.ckpt_id)
save_loss(D_loss_log,'d_loss',savedir,PARAMS.n_epoch,PARAMS.ckpt_id)  
save_loss(wd_loss_log,'wd_loss',savedir,PARAMS.n_epoch,PARAMS.ckpt_id) 
save_loss(mismatch_log,'mismatch',savedir,PARAMS.n_epoch,PARAMS.ckpt_id)


print('\n ============== DONE =================\n')



