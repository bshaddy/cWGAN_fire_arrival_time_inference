# cWGAN-UQ
Conditional Wasserstein GAN with Uncertainty Quantification 

## Creating a pytorch virtual environment using venv (https://docs.python.org/3/library/venv.html) 
Run the following commands:

```
$ python3 -m venv pytorch_env
$ source pytorch_env/bin/activate
$ pip3 install --upgrade pip
$ pip3 install matplotlib scikit-image torch h5py torchvision torchinfo
```


## Training the networks (from inside the virtual environment)
Assuming that the training npy file is available as ``Data/training_data.npy``. Run the trainining scripts from inside the ``Scripts`` folder.

To train a cWGAN network run: 

```
python3 trainer.py \
        --train_file=../Data/training_data.npy \
        --n_epoch=200 \
        --z_dim=100 \
        --n_train=8000 \
        --batch_size=5 \
        --save_freq=25 \
        --gp_coef=0.1 \
        --g_type=denseskip \
        --d_type=dense \
        --z_n_MC=5 \
        --act_param=1.0\
        --g_out=x
        --psize=512
```

The trained networks and associated results are save in exps/ 


## Continuing training from a saved model checkpoint
If one would like to continue training a model from a previous model checkpoint (helps with slurm limits on job time), run the sript training_restart.py in the ``Scripts`` folder.

To continue training from a previous checkpoint do the following (ckpt_id specifies checkpoint as epoch number to continue training from) and the model will continue training, with results added to the original training directory:

```
python3 training_restart.py \
        --train_file=../Data/training_data.npy \ 
        --GANdir=../exps/g_denseskip_x_d_dense_Nsamples8000_Ncritic4_Zdim100_BS5_Nepoch100_act_1.0_GP0.1 \
        --n_epoch=100 \
        --z_dim=100 \
        --n_train=8000 \
        --ckpt_id=100 \
        --batch_size=5 \
        --save_freq=25 \
        --gp_coef=0.1 \
        --g_type=denseskip \
        --d_type=dense \
        --z_n_MC=5 \
        --act_param=1.0 \
        --g_out=x \
	--psize=512
```



## Testing networks

You could either use a newly trained network, or the pre-trained generator checkpoint file available inside ``exps/trained_cWGAN_model_with_predictions_for_fires``. 

### Testing on fire measurement file

We demonstrate how to test the networks on the VIIRS AF composite measurement ``VIIRS_AF_measurements_and_preprocessing/Bobcat/high_confidence_Bobcat_SOD.npy``. Results are placed inside of the directory GANdir in the subdirectory called results_dir. 

For pre-trained cWGAN network

```
python3 test_on_real_fire.py \
        --GANdir=../exps/trained_cWGAN_model_with_predictions_for_fires \
        --test_file=../../VIIRS_AF_measurements_and_preprocessing/Bobcat/high_confidence_Bobcat_SOD.npy \
        --n_epoch=100 \
        --z_dim=100 \
        --n_train=8000 \
        --batch_size=5 \
        --save_freq=25 \
        --gp_coef=0.1 \
        --g_type=denseskip \
        --d_type=dense \
        --z_n_MC=200 \
        --act_param=1.0\
        --g_out=x\
        --ckpt_id=200\
        --results_dir=high_confidence_Bobcat_SOD
```


For a newly trained network (don't specify GANdir explicitly)

```
python3 test_on_real_fire.py \
        --test_file=../../VIIRS_AF_measurements_and_preprocessing/Bobcat/high_confidence_Bobcat_SOD.npy \
        --n_epoch=100 \
        --z_dim=100 \
        --n_train=8000 \
        --batch_size=5 \
        --save_freq=25 \
        --gp_coef=0.1 \
        --g_type=denseskip \
        --d_type=dense \
        --z_n_MC=200 \
        --act_param=1.0\
        --g_out=x\
        --ckpt_id=200\
        --results_dir=high_confidence_Bobcat_SOD
```

The fire arrival time results will be created and saved in the directory containing the corresponding GAN directory in ``exps``.

### Testing on samples in testing_data.npy.

For a newly trained network (don't specify GANdir explicitly)

```
python3 test_on_dataset.py \
        --test_file=../Data/testing_data.npy \
        --n_epoch=200 \
        --z_dim=100 \
        --n_train=8000 \
        --batch_size=5 \
        --save_freq=25 \
        --gp_coef=0.1 \
        --g_type=denseskip \
        --d_type=dense \
        --z_n_MC=200 \
        --act_param=1.0\
        --g_out=x\
        --ckpt_id=200\
        --n_test=5\
        --results_dir=testing_data_results
```


## The following python scripts are listed in the Scripts folder 
1. ``config.py``: Used to set read command line parameters. 
2. ``models.py``: Described the generator and various submodules used
3. ``data_utils.py``: Defines functions to manipulate images and data
4. ``utils.py``: Additional utilities 
5. ``utils_restart``: Additional utilities sprecific to the code trainer_restart.py
6. ``trainer.py``: Main training script
7. ``trainer_restart.py``: Training scripts for continuing training from a desired checkpoint. 
8. ``test_on_real_fire.py``: Main script that loads a trained network, reads a VIIRS AF composite measurement image, and performs inference of the corresponding fire arrival time. 
9. ``test_on_dataset.py``: Main script that loads a trained network, reads a data from npy file containing tau and tau_bar pairs, and performs fire arrival time inferrence on the first ``n_test`` samples.

