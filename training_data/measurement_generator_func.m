% Measurement generation function
function [upsampled_measurement] = measurement_generator_func(augmented_cropped, number_of_measurements)

counter = 0;
while counter<1

    % load fire arrival time map and downsample to VIIRS resolution
    downsampled = imresize(augmented_cropped,1/12.5,'nearest');
    meas = downsampled;

    % measurement times
    meas_t = zeros(1,number_of_measurements);
    for k = 1:number_of_measurements
        meas_t(k) = ceil(46*rand + 2);
    end
    measurement_times = sort(meas_t);

    % window criteria initialization
    window_criteria = strings(1,number_of_measurements);

    % create 4 measurements at the 4 measurement times
    for k = 1:length(measurement_times)        

        % create unique knowledge mask for each measurement band to remove
        % 50% of pixels
        kn = rand(length(meas));
        kn(kn>0.5) = 1;
        kn(kn~=1) = -1;        
        kn_meas = kn.*meas;

        % measurement window length between 6 and 12 hours 
        meas_time = measurement_times(k);
        meas_window_start = meas_time - (6*rand + 6);
        if meas_window_start<0
            meas_window_start = 0;
        end
    
        % window criteria 
        if k > 1
            window_criteria(k) = strcat('kn_meas>=', num2str(meas_window_start,'%.15f'), ' & kn_meas<=', num2str(meas_time), ' & kn_meas~=', num2str(measurement_times(k-1)));
        else 
            window_criteria(k) = strcat('kn_meas>=', num2str(meas_window_start,'%.15f'), ' & kn_meas<=', num2str(meas_time));
        end
      
        % make  measurement, allowing for overlap of measurement windows
        meas(eval(window_criteria(k))) = meas_time;
    end

    % set unburnt pixels
    last_meas_time = meas_time;
    meas(meas>last_meas_time) = max(meas,[],"all");        % unburnt pixels set to max meas time 
    
    % set window space pixels
    window_spaces_criteria = strcat('meas~=', num2str(measurement_times(1)));
    for i = 2:number_of_measurements
        window_spaces_criteria = strcat(window_spaces_criteria, ' & meas~=', num2str(measurement_times(i)));
    end
    window_spaces_criteria = strcat(window_spaces_criteria, ' & meas~=', num2str(max(meas,[],"all"),'%.15f'));
    meas(eval(window_spaces_criteria)) = max(meas,[],"all");   

    % eliminate 3 random 8x8 pixel patches (3000x3000m)  
    measurement_interim = meas;
    for k = 1:3
        box_row_start = ceil(75*rand);
        box_col_start = ceil(75*rand);
        measurement_interim(box_row_start:box_row_start+7,box_col_start:box_col_start+7) = max(measurement_interim,[],"all");
    end

    % make sure there are at least 4 fire PIXELS after applying knowledge mask
    if length(measurement_interim(measurement_interim~=max(measurement_interim,[],"all"))) >= 4  % can also use length(unique(measurement_interim)) == 6 to make sure all 4 measurement TIMES are represented
        measurement = measurement_interim;

        % upsampled measurement
        upsampled_measurement = imresize(measurement,512/82,'nearest');                          % upsample back to original resolution

        % filter upsampled measurement to make sure there are no unwanted
        % pixel values
        upsample_filter_criteria = strcat('upsampled_measurement~=', num2str(measurement_times(1)));
        for i = 2:number_of_measurements
            upsample_filter_criteria = strcat(upsample_filter_criteria, ' & upsampled_measurement~=', num2str(measurement_times(i)));
        end
        upsample_filter_criteria = strcat(upsample_filter_criteria, ' & upsampled_measurement~=', num2str(max(upsampled_measurement,[],"all")));
        upsampled_measurement(eval(upsample_filter_criteria)) = max(upsampled_measurement,[],"all");

        % normalize
        upsampled_measurement = upsampled_measurement/max(upsampled_measurement,[],'all');
        
        counter = counter + 1;
    end    
end

end