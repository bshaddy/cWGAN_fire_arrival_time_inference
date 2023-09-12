This directory contains the formatted training and testing datasets used for this work. 

training_data.npy and testing_data.npy contain training and testing data for the version of the fire arrival time inference problem where ignition time is know (i.e. fire arrival times in these datasets are relative to a known ignition time). 

training_data_random_start.npy and testing_data_random_start.npy contain the training and testing data for the version of the fire arrival time inference problem where ignition time in not know (i.e. a random amount of time has been added to all fire arrival times in these datasets to account for an unknown ignition time). These versions of the training and testing data have been used to train the version of the cWGAN which predicts fire arrival times relative to the start of the day and therefore also predicts ignition time, as presented in the relevant manuscript.   
