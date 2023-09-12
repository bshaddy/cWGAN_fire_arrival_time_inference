All necessary input files for running the idealized WRF-SFIRE simulations used to create training data for the cWGAN-based method for inferring fire arrival times from satellite measurements are included in this directory.

namelist.input_training_data_simulations is used for all simulations.
namelist.fire_training_data_simulations is also used for all simulations.

Atmospheric soundings used for the initial atmospheric state vary accross all simulations. Atmospheric sounding input files are named as follows: input_sounding_<case_number>_u_10_<u10_wind_speed>
