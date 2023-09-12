% Training data generation
clear; clc; close all;

files = {'fire_arrival_time_case_1_u_10_3.27039549238391'   'fire_arrival_time_case_2_u_10_3.44607251570004'    'fire_arrival_time_case_3_u_10_3.74075796411855'    'fire_arrival_time_case_4_u_10_2.25270799251249' ...
    'fire_arrival_time_case_5_u_10_0.419106889984663'   'fire_arrival_time_case_6_u_10_1.14488484358409'    'fire_arrival_time_case_7_u_10_4.56668680750835'    'fire_arrival_time_case_8_u_10_0.761890094846115' ...
    'fire_arrival_time_case_9_u_10_4.12908488744774'    'fire_arrival_time_case_10_u_10_2.69171217630029'   'fire_arrival_time_case_11_u_10_4.98067358313443'   'fire_arrival_time_case_12_u_10_0.390877643765918' ...
    'fire_arrival_time_case_13_u_10_2.21339134887723'   'fire_arrival_time_case_14_u_10_0.533263850902922'  'fire_arrival_time_case_15_u_10_4.80949040427527'   'fire_arrival_time_case_16_u_10_0.0231711206703372' ...
    'fire_arrival_time_case_17_u_10_3.87455232355751'   'fire_arrival_time_case_18_u_10_4.08651610326717'   'fire_arrival_time_case_19_u_10_4.34347352681755'   'fire_arrival_time_case_20_u_10_0.422179227554552'};

number_of_measurements = 4;  % desired number of measurement times to use

for Case=1:20
    dir_to_fire_spread_solutions = 'WRF-SFIRE_solutions';                             % directory containing WRF-SFIRE solutions
    filename = strcat(dir_to_fire_spread_solutions,'\',char(files(Case)),'.mat');
    load(filename);

    for augment=1:500
        % augmentation
        rotated = imrotate(fire_arrival_time,360*rand,'nearest','crop');              % rotates randomly between 0 and 360 deg
        translated = imtranslate(rotated,[300*rand-150, 300*rand-150],'nearest');     % translate within a box of 9000x9000m (i.e. +-150 grid spaces x 30m grid spacing)
        translated(translated == 0) = max(translated,[],'all');
        augmented = translated;

        % crop, downsample, and normalize
        augmented_cropped = augmented(124:end-123,124:end-123)/3600;
        augmented_cropped_downsampled = imresize(augmented_cropped,0.5);
        fire_arrival_time_map = augmented_cropped_downsampled;
        dir_for_augmented_arrival_time_maps = 'augmented_fire_arrival_time_maps';     % directory where augmented fire arrival time maps will be placed 
        if Case == 1 && augment == 1
            mkdir(dir_for_augmented_arrival_time_maps);
        end
        filename_fire_arrival_time_map = strcat(dir_for_augmented_arrival_time_maps,'\',char(files(Case)), ... 
            '_augmented_cropped_downsampled_',num2str(augment));
        save(filename_fire_arrival_time_map,'fire_arrival_time_map');

        % measurement 
        upsampled_measurement = measurement_generator_func(augmented_cropped, number_of_measurements); 
%         number_of_pixel_values = unique(upsampled_measurement)

        % upsampled measurement
        dir_for_augmented_arrival_time_map_measurements = 'augmented_fire_arrival_time_map_measurements';                % directory where measurements corresponding to augmented fire arrival time maps will be placed 
        if Case == 1 && augment == 1
            mkdir(dir_for_augmented_arrival_time_map_measurements);
        end
        filename_upsampled_measurement = strcat(dir_for_augmented_arrival_time_map_measurements,'\',char(files(Case)), ... 
            '_augmented_cropped_downsampled_',num2str(augment),'_measurement_upsampled');
        save(filename_upsampled_measurement,'upsampled_measurement');

    end
end