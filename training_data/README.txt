This directory contains the following items for use in generating training data for the problem of inferring fire arrival times from satellite measurements using a cWGAN:

Subdirectory WRF-SFIRE_input_files which contains the WRF-SFIRE input files used to run 20 idealized WRF_SFIRE simulations, generating the 20 fire spread solutions used to produce training data here.

Subdirectory WRF-SFIRE_solutions which contains the fire arrival time maps solutions for the 20 fire simulations in .mat format.

MATLAB script training_data_generation.m which augments the 20 WRF-SFIRE solutions contained in subdirectory WRF-SFIRE_solutions by rotating and translating, producing 10,000 instances of fire arrival time maps, which are placed into the subdirectory augmented_fire_arrival_time_maps that is created by this script. 

MATLAB function measurement_geneator_func.m which is used in script training_data_generation.m to produce a measurement corresponding to each fire arrival time map. Measurements are placed into subdirectory augmented_fire_arrival_time_map_measurements which is created by training_data_generation.m.

Python notebook training_data_formater.ipynb which creates the training and testing dataset files training_data.npy and testing_data.npy using the data in subdirectories augmented_fire_arrival_time_maps and augmented_fire_arrival_time_map_measurements and places them into the subdirectory final_training_and_testing_datasets.

Python notebook training_data_formater_random_start_time.ipynb which creates the training and testing dataset files training_data_random_start.npy and testing_data_random_start.npy using the files training_data.npy and testing_data.npy as a starting point. training_data_random_start.npy and testing_data_random_start.npy are placed into the subrirectory final_training_and_testing_datasets. 

Subdirectory final_training_and_testing_datasets contains compressed versions of the training and testing dataset files training_data.npy, testing_data.npy, training_data_random_start.npy and testing_data_random_start.npy. The training and testing datasets used for the relevant manuscript, wherein fire arrival times are predicted relative to the start of the day based on satellite measurements and ignition time is unknown, are training_data_random_start.npy and testing_data_random_start.npy.
