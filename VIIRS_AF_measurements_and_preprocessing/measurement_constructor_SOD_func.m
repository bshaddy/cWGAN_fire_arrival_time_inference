% Function for creating composite measurements out of VIIRS AF data
% Input: center lat and lon of fire, ignitinon day as Julian day, 
% directory containing measurement files, confidence level cutoff (7=low conf, 8=nominal conf, 9=high conf), 
% clean up mode (0=don't do anything, 1=delete unused VIIRS measurement files)

function measurement = measurement_constructor_SOD_func(center_lat,center_lon,ign_day,measurement_file_directory,conf_level,clean_up_mode)

% Create lat-lon discretization
x = 0:375:375*82;
y = x;
z = zeros(length(x));
bottom_left_lat = center_lat - 0.12;
bottom_left_lon = center_lon - 0.12;
origin = [bottom_left_lat,bottom_left_lon,0];
[lat,lon] = local2latlon(x,y,z,origin);

lat_nodes = rot90(lat);
lon_nodes = lon;

fprintf('Min Latitude = %f; ',min(lat,[],'all'));
fprintf('Max Latitude = %f; ',max(lat,[],'all'));
fprintf('Min Longitude = %f; ',min(lon,[],'all'));
fprintf('Max Longitude = %f\n\n',max(lon,[],'all'));

% Untar any .tar measurement files in measurement file directory
measurement_files_tar = strcat(measurement_file_directory,'\*.tar');
dinfo_tar = dir(measurement_files_tar);
filenames_tar = {dinfo_tar.name};
for k = 1:length(filenames_tar)
    untar_file = strcat(measurement_file_directory,'\',filenames_tar{k});
    untar(untar_file,measurement_file_directory);
end

% Get filenames
measurement_files = strcat(measurement_file_directory,'\*.nc');
dinfo = dir(measurement_files);
filenames = {dinfo.name};

% Load measurements and sort AF detections into lat-lon discretization
arrival_times = zeros(82,82);
for f = 1:length(filenames)

    % Current measurement file
    meas_file = filenames{f};
    file = strcat(measurement_file_directory,'\',meas_file);

    % Read in AF locations and confidence levels
    if length(meas_file)==71                               % Read NOAA measurements
        FP_lon = ncread(file,'/Fire Pixels/FP_longitude');
        FP_lat = ncread(file,'/Fire Pixels/FP_latitude');    
        confidence = ncread(file,'/Fire Pixels/FP_confidence');
    elseif length(meas_file)==43 || length(meas_file)==47  % Read NASA measurements (including NRT)
        FP_lon = ncread(file,'FP_longitude');
        FP_lat = ncread(file,'FP_latitude');
        confidence = ncread(file,'FP_confidence');
    end
    fprintf('%s Loaded\n',meas_file);

    % Read measurement time from filename
    if length(meas_file)==71                                                                                     % Read NOAA measurement time
        t = datetime(str2double(meas_file(20:23)),str2double(meas_file(24:25)),str2double(meas_file(26:27)));
        meas_day = day(t,'dayofyear');
        meas_hour = str2double(meas_file(28:29));
        meas_min = str2double(meas_file(30:31));
    elseif length(meas_file)==43                                                                                 % Read NASA measurement time 
        meas_day = str2double(meas_file(15:17));
        meas_hour = str2double(meas_file(19:20));
        meas_min = str2double(meas_file(21:22));
    elseif length(meas_file)==47                                                                                 % Read NASA NRT measurement time
        meas_day = str2double(meas_file(19:21));
        meas_hour = str2double(meas_file(23:24));
        meas_min = str2double(meas_file(25:26));
    end
    
    % Compute arrival time
    meas_tot_hour = meas_hour + meas_min/60;
    day_diff = meas_day - ign_day;
    arr_time = 24*day_diff + meas_tot_hour;
    fprintf('Arrival time = %f\n',arr_time);

    % Create arrival time measurement map  
    FP_used = 0;
    for k = 1:length(FP_lon)
        lon_current = FP_lon(k);
        lat_current = FP_lat(k);
        conf = confidence(k);
        for i = 1:82
            for j = 1:82
                if lon_current>=lon_nodes(1,j) && lon_current<lon_nodes(1,j+1) && lat_current<=lat_nodes(i,1) && lat_current>lat_nodes(i+1,1) && (arrival_times(i,j)==0 || arrival_times(i,j)>arr_time) && arr_time<72 && arr_time>0 && conf>=conf_level
                    arrival_times(i,j) = arr_time;
                    FP_used = FP_used + 1;               
                end
            end
        end
    end
    fprintf('%.0f fire pixels activated\n',FP_used);

    % If clean_up_mode is set to 1, unused files will be deleted.
    % If clean_up_mode is set to 0, nothing happens. 
    switch clean_up_mode
        case 1
            if FP_used==0
                delete(file);
                fprintf('File %s deleted\n\n',meas_file);
            else
                fprintf('File %s retained\n\n',meas_file);
            end

        case 0
            if FP_used==0
                fprintf('File %s unused\n\n',meas_file);
            else
                fprintf('File %s used\n\n',meas_file);
            end
    end

end

% set background pixel values, adjust resolution to cWGAN resolution, normalize
arrival_times(arrival_times==0) = 72;
arrival_times = imresize(arrival_times,512/82,'nearest');
measurement = arrival_times/max(arrival_times,[],'all');

% Plot arrival time map
imshow(measurement,colormap=hot);
ax = gca;
ax.CLim = [0 1.015];

if conf_level == 8
    conf_str = 'high+nominal';
elseif conf_level == 9
    conf_str = 'high';
elseif conf_level == 7
    conf_str = 'high+nominal+low';
end

measurement_save_file = strcat(measurement_file_directory,"\" +conf_str+ "_confidence_measurement_from_SOD.mat");
save(measurement_save_file,'measurement');

disp('Success!!!');

end

