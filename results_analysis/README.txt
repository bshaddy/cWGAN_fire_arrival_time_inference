This directory contains the following data and codes related to evaluating the performance of the trained cWGAN in predicting fire arrival times from VIIRS AF measurements:

Subdirectory IR_perimeters contains the airborne IR perimeters measured for the fires considered here, which are used for ground truth comparisons in the associated manuscript.

Subdirectory SVM_predictions contains the predictions of fire arrival times produced by the SVM method for the VIIRS AF measurements considered in the assocaiated manuscript.

MATLAB script Results_Analysis_and_Plots performs the computation of predicted ignition times, SC, POD, and FAR for the cWGAN and SVM methods, in addition to producing the result plots preseted in the manuscript. The cWGAN predictions used for these computations are gathered from the cWGAN_codes directory, which contains the trained model in the exps folder. Predicted fire arrival time maps are contained within folders in the trained cWGAN model folder.
