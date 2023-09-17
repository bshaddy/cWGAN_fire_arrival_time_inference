Contained within this repository are the data and scripts needed for performing inference of fire arrival times from VIIRS AF satellite measurements utilizing a cWGAN, as described in the manuscript Generative Algorithms for Fusion of Physics-Based Wildfire Spread Models with Satellite Data for Initializing Wildfire Forecasts (https://doi.org/10.48550/arXiv.2309.02615).

Folder training_data contains all information needed to create the training data used here, in addition to the specific pieces of data which have been utilized.

Folder cWGAN_codes contains all of the codes pertaining to the conditional Wasserstein GAN utilized for this work. The trained network used for the results presented in the accompanying work is contained within the exps subfolder of this directory. Resulting fire arrival time samples inferred from VIIRS AF measurements by the trained cWGAN, which have been used to evaluate the presented method, are also contained within the folder for the trained model.

Folder VIIRS_AF_measurements_and_preprocessing contains all preprocessing codes and data pertaining to the VIIRS AF measurements that have been used to evaluate the proposed method. 

Folder results_analysis contains the codes and data pertaining to evaluation of the performance of the cWGAN method and comparisons made to the SVM method and to the groud-truth IR fire perimeter measurements. The code for producing result figures is also included. 

Additional information on the contents of each folders is provided in the README files contained within the pertinent folder. 
