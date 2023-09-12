# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Lambda
from functools import partial
from scipy.linalg import qr
import time,random
from config import cla
from models import *
from data_utils import *
from utils import *
from plot_dataset_results_real_fire import plot_results

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

PARAMS = cla()

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)

print('\n ============== LOADING DE-RAINER =================\n')

print('\n --- Loading data from file\n')

test_data    = LoadDataset(datafile   = PARAMS.test_file,
                           N          = PARAMS.n_test,
                           permute    = True)
n_test,x_C,x_W,x_H = test_data.x_shape

print('\n --- Loading conditional GAN generator from checkpoint\n')
if PARAMS.GANdir == None:
    GANdir = get_GAN_dir(PARAMS)
else:
    GANdir = PARAMS.GANdir    

if PARAMS.g_type == 'resskip':
    G_model = generator_resskip(x_shape=(x_C,x_H,x_W),z_dim=PARAMS.z_dim,g_out=PARAMS.g_out,act_param=PARAMS.act_param)  
elif PARAMS.g_type == 'denseskip':
    G_model = generator_denseskip(x_shape=(x_C,x_H,x_W),z_dim=PARAMS.z_dim,denselayers=3,g_out=PARAMS.g_out,act_param=PARAMS.act_param)    

if PARAMS.z_dim == None:
    z_n_MC = 1
    ncol   = 3
    qoi    = 'mode'
else:    
    z_n_MC = PARAMS.z_n_MC
    ncol = 4
    qoi    = 'mean'
G_state  = torch.load(f"{GANdir}/checkpoints/G_model_{PARAMS.ckpt_id}.pth",map_location=torch.device(device))
G_model.load_state_dict(G_state)
G_model.eval()  # NOTE: switching to eval mode for generator

# Creating sub-function to generate latent variables
glv          = partial(get_lat_var,x_W=x_W,x_H=x_H,z_dim=PARAMS.z_dim)
nimp_samples = 3 # Used only when z_dim > 0

# Creating results directory
results_dir = f"{GANdir}/{PARAMS.results_dir}"
print(f"\n --- Generating results directory: {results_dir}")
if os.path.exists(results_dir):
        print('\n     *** Folder already exists!\n')    
else:
    os.makedirs(results_dir)

print(f"\n --- Computing statistics\n")

for n in range(n_test):
    
    test_x,test_y = test_data[n:n+1]

    print(f"\n ---   for test sample {n}\n")

    if PARAMS.z_dim is not None:
        sample_mean  = np.zeros((x_W,x_H))     
        sample_var   = np.zeros((x_W,x_H))
        snapshot_mat = np.zeros((x_H*x_W,z_n_MC)) 
        with torch.no_grad():
            G_model.eval()  # NOTE: switching to eval mode for generator 
            predictions = np.zeros((z_n_MC,PARAMS.psize,PARAMS.psize))
            for i in range(z_n_MC):
                test_z = glv(batch_size=1)
                pred_  = G_model(test_y,test_z).detach().numpy().squeeze()
                
                predictions[i,:,:] = pred_

                snapshot_mat[:,i] = np.reshape(pred_,(x_H*x_W))
                old_mean    = np.copy(sample_mean)
                sample_mean = sample_mean + (pred_-sample_mean)/(i+1)
                sample_var  = sample_var  + (pred_-sample_mean)*(pred_-old_mean)
            sample_var /= (z_n_MC-1)
            sample_std = np.sqrt(sample_var)
            np.save(f"{results_dir}/samples.npy",predictions)

        print(f"     ... performing rrqr")
        _,_,p = qr(snapshot_mat,pivoting=True, mode='economic')
        imp_samples = np.reshape(snapshot_mat[:,p[0:nimp_samples]].T,(-1,x_W,x_H))     

        print(f"     ... saving stats")
        save_data = np.stack((test_y.detach().numpy().squeeze(),
                              sample_mean,
                              sample_std),axis=0)
        save_data = np.concatenate((save_data,imp_samples),axis=0)
    else:
        with torch.no_grad():  
            test_z = glv(batch_size=1)
            pred_  = G_model(test_y,test_z).detach().numpy().squeeze()

        print(f"     ... saving stats")
        save_data = np.stack((test_y.detach().numpy().squeeze(),
                              pred_),axis=0)
            

    np.save(f"{results_dir}/stats_sample_{n+1}_ckpt_{PARAMS.ckpt_id}.npy",save_data)

print(f"\n --- Generating plots")
plot_results(PARAMS)



print("----------------------- DONE --------------------------")



 
