clear; clc; close all;
weight = 0.2;

%% cWGAN Statistics Figure

% Load cWGAN results
[high_conf_samples_bobcat,high_nom_conf_samples_bobcat,weighted_mean_bobcat,weighted_SD_bobcat,weighted_mean_plus_SD_bobcat,weighted_mean_minus_SD_bobcat] = load_GAN_samples_mixture_model("..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high_confidence_bobcat_SOD\samples.mat", ...
                                                                                                                                    "..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high+nominal_confidence_bobcat_SOD\samples.mat", ...
                                                                                                                                    weight);
[high_conf_samples_tennant,high_nom_conf_samples_tennant,weighted_mean_tennant,weighted_SD_tennant,weighted_mean_plus_SD_tennant,weighted_mean_minus_SD_tennant] = load_GAN_samples_mixture_model("..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high_confidence_Tennant_SOD\samples.mat", ...
                                                                                                                                    "..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high+nominal_confidence_Tennant_SOD\samples.mat", ...
                                                                                                                                    weight);
[high_conf_samples_oak,high_nom_conf_samples_oak,weighted_mean_oak,weighted_SD_oak,weighted_mean_plus_SD_oak,weighted_mean_minus_SD_oak] = load_GAN_samples_mixture_model("..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high_confidence_Oak_SOD\samples.mat", ...
                                                                                                                                    "..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high+nominal_confidence_Oak_SOD\samples.mat", ...
                                                                                                                                    weight);
[high_conf_samples_mineral,high_nom_conf_samples_mineral,weighted_mean_mineral,weighted_SD_mineral,weighted_mean_plus_SD_mineral,weighted_mean_minus_SD_mineral] = load_GAN_samples_mixture_model("..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high_confidence_Mineral_SOD\samples.mat", ...
                                                                                                                                    "..\cWGAN_codes\exps\trained_cWGAN_model_with_predictions_for_fires\high+nominal_confidence_Mineral_SOD\samples.mat", ...
                                                                                                                                    weight);

% Determine geolocation for cWGAN predictions
[lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat] = GAN_geolocation(34.259,-117.955);
[lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant] = GAN_geolocation(41.665191,-122.054254);
[lat_oak,lon_oak,latlim_oak,lonlim_oak] = GAN_geolocation(37.5509366,-119.9234728);
[lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral] = GAN_geolocation(36.184,-120.5568);

% cWGAN Plots
f = figure;
tiledlayout(2,4,"TileSpacing","compact")
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_bobcat,hot,1,lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_tennant,hot,1,lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_oak,hot,1,lat_oak,lon_oak,latlim_oak,lonlim_oak,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_mineral,hot,1,lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral,10,1);
nexttile
make_prediction_statistics_plot_for_fig(weighted_SD_bobcat,parula,0,lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_SD_tennant,parula,0,lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_SD_oak,parula,0,lat_oak,lon_oak,latlim_oak,lonlim_oak,10,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_SD_mineral,parula,0,lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral,10,1);
f.WindowState = 'maximized';
% exportgraphics(gcf,"cWGAN_prediction_statistics_figure.jpg","Resolution",1000);


%% SVM fire arrival time predictions

% Load SVM comparison
[lat_SVM_bobcat,lon_SVM_bobcat,tign_SVM_bobcat] = load_SVM("SVM_predictions\tign_bobcat.nc");
[lat_SVM_tennant,lon_SVM_tennant,tign_SVM_tennant] = load_SVM("SVM_predictions\tign_tennant.nc");
[lat_SVM_oak,lon_SVM_oak,tign_SVM_oak] = load_SVM("SVM_predictions\tign_oak.nc");
[lat_SVM_mineral,lon_SVM_mineral,tign_SVM_mineral] = load_SVM("SVM_predictions\tign_mineral.nc");

% SVM postprocess
tign_SVM_bobcat(tign_SVM_bobcat == mode(tign_SVM_bobcat)) = 72/24;
tign_SVM_tennant(tign_SVM_tennant == mode(tign_SVM_tennant)) = 72/24;
tign_SVM_oak(tign_SVM_oak == mode(tign_SVM_oak)) = 72/24;
tign_SVM_mineral(tign_SVM_mineral == mode(tign_SVM_mineral)) = 72/24;

% SVM Plots
f = figure;
tiledlayout(1,4,"TileSpacing","compact")
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_bobcat,hot,1,lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat,20,0);
make_prediction_statistics_plot_for_fig(tign_SVM_bobcat*24/72,hot,1,lat_SVM_bobcat,lon_SVM_bobcat,latlim_bobcat,lonlim_bobcat,20,0);
nexttile
make_prediction_statistics_plot_for_fig(weighted_mean_tennant,hot,1,lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant,20,0);
make_prediction_statistics_plot_for_fig(tign_SVM_tennant*24/72,hot,1,lat_SVM_tennant,lon_SVM_tennant,latlim_tennant,lonlim_tennant,20,0);
nexttile 
make_prediction_statistics_plot_for_fig(weighted_mean_oak,hot,1,lat_oak,lon_oak,latlim_oak,lonlim_oak,20,0);
make_prediction_statistics_plot_for_fig(tign_SVM_oak*24/72,hot,1,lat_SVM_oak,lon_SVM_oak,latlim_oak,lonlim_oak,20,0);
nexttile 
make_prediction_statistics_plot_for_fig(weighted_mean_mineral,hot,1,lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral,20,1);
make_prediction_statistics_plot_for_fig(tign_SVM_mineral*24/72,hot,1,lat_SVM_mineral,lon_SVM_mineral,latlim_mineral,lonlim_mineral,20,1);
f.WindowState = 'maximized';
% exportgraphics(gcf,"prediction_statistics_figure_SVM.jpg","Resolution",1000);


%% Sorensen's coefficient computation

% Load IR perimeter polyshape
IR_perim_bobcat = load_IR("IR_perimeters\bobcat\20200908_0115_bobcat_HeatPerimeter.shp",0,1);
IR_perim_tennant = load_IR("IR_perimeters\Tennant\20210629_2305_Tennant_HeatPerimeter.shp",1,1);
IR_perim_oak = load_IR("IR_perimeters\Oak\20220723_2246_PDT_Oak_HeatPerimeter.shp",1,1);
IR_perim_mineral = load_IR("IR_perimeters\Mineral\20200714_2015_Mineral_HeatPerimeter.shp",0,1);

% Create polygons from geolocated rasters cWGAN
gan_poly_bobcat = create_poly_from_raster(lat_bobcat,lon_bobcat,weighted_mean_bobcat*72,56.25);
gan_poly_tennant = create_poly_from_raster(lat_tennant,lon_tennant,weighted_mean_tennant*72,54.08);
gan_poly_oak = create_poly_from_raster(lat_oak,lon_oak,weighted_mean_oak*72,53.75);
gan_poly_mineral = create_poly_from_raster(lat_mineral,lon_mineral,weighted_mean_mineral*72,51.25);

% Create polygons from geolocated rasters SVM
svm_poly_bobcat = create_poly_from_raster(lat_SVM_bobcat,lon_SVM_bobcat,tign_SVM_bobcat*24,56.25);
svm_poly_tennant = create_poly_from_raster(lat_SVM_tennant,lon_SVM_tennant,tign_SVM_tennant*24,54.08);
svm_poly_oak = create_poly_from_raster(lat_SVM_oak,lon_SVM_oak,tign_SVM_oak*24,53.75);
svm_poly_mineral = create_poly_from_raster(lat_SVM_mineral,lon_SVM_mineral,tign_SVM_mineral*24,51.25);

% Sorensen / POD / FAR calculations cWGAN
[SC_gan_bobcat,POD_gan_bobcat,FAR_gan_bobcat,A_gan_bobcat,B_gan_bobcat,C_gan_bobcat] = area_discrepancy_calculations(IR_perim_bobcat,gan_poly_bobcat);
[SC_gan_tennant,POD_gan_tennant,FAR_gan_tennant,A_gan_tennant,B_gan_tennant,C_gan_tennant] = area_discrepancy_calculations(IR_perim_tennant,gan_poly_tennant);
[SC_gan_oak,POD_gan_oak,FAR_gan_oak,A_gan_oak,B_gan_oak,C_gan_oak] = area_discrepancy_calculations(IR_perim_oak,gan_poly_oak);
[SC_gan_mineral,POD_gan_mineral,FAR_gan_mineral,A_gan_mineral,B_gan_mineral,C_gan_mineral] = area_discrepancy_calculations(IR_perim_mineral,gan_poly_mineral);

% Sorensen / POD / FAR calculations SVM
[SC_svm_bobcat,POD_svm_bobcat,FAR_svm_bobcat,A_svm_bobcat,B_svm_bobcat,C_svm_bobcat] = area_discrepancy_calculations(IR_perim_bobcat,svm_poly_bobcat);
[SC_svm_tennant,POD_svm_tennant,FAR_svm_tennant,A_svm_tennant,B_svm_tennant,C_svm_tennant] = area_discrepancy_calculations(IR_perim_tennant,svm_poly_tennant);
[SC_svm_oak,POD_svm_oak,FAR_svm_oak,A_svm_oak,B_svm_oak,C_svm_oak] = area_discrepancy_calculations(IR_perim_oak,svm_poly_oak);
[SC_svm_mineral,POD_svm_mineral,FAR_svm_mineral,A_svm_mineral,B_svm_mineral,C_svm_mineral] = area_discrepancy_calculations(IR_perim_mineral,svm_poly_mineral);


%% Ignition time computations

% Ignition times cWGAN
ignition_time_gan_bobcat = min(weighted_mean_bobcat,[],'all')*72;
ignition_time_gan_tennant = min(weighted_mean_tennant,[],'all')*72;
ignition_time_gan_oak = min(weighted_mean_oak,[],'all')*72;
ignition_time_gan_mineral = min(weighted_mean_mineral,[],'all')*72;

% Ignition times SVM
ignition_time_svm_bobcat = min(tign_SVM_bobcat,[],'all')*24;
ignition_time_svm_tennant = min(tign_SVM_tennant,[],'all')*24;
ignition_time_svm_oak = min(tign_SVM_oak,[],'all')*24;
ignition_time_svm_mineral = min(tign_SVM_mineral,[],'all')*24;


%% Spatial discrepancy figure
f = figure;

tiledlayout(2,4,"TileSpacing","tight")
nexttile
make_area_discrepancy_subplots(A_gan_bobcat,B_gan_bobcat,C_gan_bobcat,'bobcat cWGAN vs IR Perimeter',lat_bobcat,lon_bobcat,[latlim_bobcat(1)+0.02 latlim_bobcat(2)-0.08],[lonlim_bobcat(1)+0.02 lonlim_bobcat(2)-0.08],weighted_mean_bobcat,12);
nexttile
make_area_discrepancy_subplots(A_gan_tennant,B_gan_tennant,C_gan_tennant,'Tennant cWGAN vs IR Perimeter',lat_tennant,lon_tennant,[latlim_tennant(1)+0.07 latlim_tennant(2)-0.03],[lonlim_tennant(1)+0.01 lonlim_tennant(2)-0.09],weighted_mean_tennant,12);
nexttile
make_area_discrepancy_subplots(A_gan_oak,B_gan_oak,C_gan_oak,'Oak cWGAN vs IR Perimeter',lat_oak,lon_oak,[latlim_oak(1)+0.02 latlim_oak(2)-0.08],[lonlim_oak(1)+0.05 lonlim_oak(2)-0.05],weighted_mean_oak,12);
nexttile
make_area_discrepancy_subplots(A_gan_mineral,B_gan_mineral,C_gan_mineral,'Mineral cWGAN vs IR Perimeter',lat_mineral,lon_mineral,[latlim_mineral(1)+0.02 latlim_mineral(2)-0.08],[lonlim_mineral(1)+0.03 lonlim_mineral(2)-0.07],weighted_mean_mineral,12);
nexttile
make_area_discrepancy_subplots(A_svm_bobcat,B_svm_bobcat,C_svm_bobcat,'bobcat cWGAN vs IR Perimeter',lat_bobcat,lon_bobcat,[latlim_bobcat(1)+0.02 latlim_bobcat(2)-0.08],[lonlim_bobcat(1)+0.02 lonlim_bobcat(2)-0.08],weighted_mean_bobcat,12);
nexttile
make_area_discrepancy_subplots(A_svm_tennant,B_svm_tennant,C_svm_tennant,'Tennant cWGAN vs IR Perimeter',lat_tennant,lon_tennant,[latlim_tennant(1)+0.07 latlim_tennant(2)-0.03],[lonlim_tennant(1)+0.01 lonlim_tennant(2)-0.09],weighted_mean_tennant,12);
nexttile
make_area_discrepancy_subplots(A_svm_oak,B_svm_oak,C_svm_oak,'Oak cWGAN vs IR Perimeter',lat_oak,lon_oak,[latlim_oak(1)+0.02 latlim_oak(2)-0.08],[lonlim_oak(1)+0.05 lonlim_oak(2)-0.05],weighted_mean_oak,12);
nexttile
make_area_discrepancy_subplots(A_svm_mineral,B_svm_mineral,C_svm_mineral,'Mineral cWGAN vs IR Perimeter',lat_mineral,lon_mineral,[latlim_mineral(1)+0.02 latlim_mineral(2)-0.08],[lonlim_mineral(1)+0.03 lonlim_mineral(2)-0.07],weighted_mean_mineral,12);
f.WindowState = 'maximized';
% exportgraphics(gcf,"area_discrepancy_figure.jpg","Resolution",1000);


%% VIIRS measurements figure

% Load measurements
[high_conf_meas_bobcat,high_nom_conf_meas_bobcat] = load_measurement_images("..\VIIRS_AF_measurements_and_preprocessing\bobcat\high_confidence_measurement_from_SOD.mat", ...
                                                              "..\VIIRS_AF_measurements_and_preprocessing\bobcat\high+nominal_confidence_measurement_from_SOD.mat");
[high_conf_meas_tennant,high_nom_conf_meas_tennant] = load_measurement_images("..\VIIRS_AF_measurements_and_preprocessing\Tennant\high_confidence_measurement_from_SOD.mat", ...
                                                              "..\VIIRS_AF_measurements_and_preprocessing\Tennant\high+nominal_confidence_measurement_from_SOD.mat");
[high_conf_meas_oak,high_nom_conf_meas_oak] = load_measurement_images("..\VIIRS_AF_measurements_and_preprocessing\Oak\high_confidence_measurement_from_SOD.mat",...
                                                              "..\VIIRS_AF_measurements_and_preprocessing\Oak\high+nominal_confidence_measurement_from_SOD.mat");
[high_conf_meas_mineral,high_nom_conf_meas_mineral] = load_measurement_images("..\VIIRS_AF_measurements_and_preprocessing\Mineral\high_confidence_measurement_from_SOD.mat",...
                                                              "..\VIIRS_AF_measurements_and_preprocessing\Mineral\high+nominal_confidence_measurement_from_SOD.mat");

% Load IR perimeter geoshape
IR_perim_bobcat = load_IR("IR_perimeters\bobcat\20200908_0115_bobcat_HeatPerimeter.shp",0,0);
IR_perim_tennant = load_IR("IR_perimeters\Tennant\20210629_2305_Tennant_HeatPerimeter.shp",1,0);
IR_perim_oak = load_IR("IR_perimeters\Oak\20220723_2246_PDT_Oak_HeatPerimeter.shp",1,0);
IR_perim_mineral = load_IR("IR_perimeters\Mineral\20200714_2015_Mineral_HeatPerimeter.shp",0,0);

% Determine geolocation for GAN predictions
[lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat] = GAN_geolocation(34.259,-117.955);
[lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant] = GAN_geolocation(41.665191,-122.054254);
[lat_oak,lon_oak,latlim_oak,lonlim_oak] = GAN_geolocation(37.5509366,-119.9234728);
[lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral] = GAN_geolocation(36.184,-120.5568);

% plots
f = figure;
tiledlayout(2,4,"TileSpacing","compact")
nexttile
make_measurement_plot_for_figure(high_conf_meas_bobcat,IR_perim_bobcat,lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat,10,1,0);
nexttile
make_measurement_plot_for_figure(high_conf_meas_tennant,IR_perim_tennant,lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant,10,1,0);
nexttile
make_measurement_plot_for_figure(high_conf_meas_oak,IR_perim_oak,lat_oak,lon_oak,latlim_oak,lonlim_oak,10,1,0);
nexttile
make_measurement_plot_for_figure(high_conf_meas_mineral,IR_perim_mineral,lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral,10,1,1);
nexttile
make_measurement_plot_for_figure(high_nom_conf_meas_bobcat,IR_perim_bobcat,lat_bobcat,lon_bobcat,latlim_bobcat,lonlim_bobcat,10,1,0);
nexttile
make_measurement_plot_for_figure(high_nom_conf_meas_tennant,IR_perim_tennant,lat_tennant,lon_tennant,latlim_tennant,lonlim_tennant,10,1,0);
nexttile
make_measurement_plot_for_figure(high_nom_conf_meas_oak,IR_perim_oak,lat_oak,lon_oak,latlim_oak,lonlim_oak,10,1,0);
nexttile
make_measurement_plot_for_figure(high_nom_conf_meas_mineral,IR_perim_mineral,lat_mineral,lon_mineral,latlim_mineral,lonlim_mineral,10,1,1);
f.WindowState = 'maximized';
% exportgraphics(gcf,"measurement_plots_figure.jpg","Resolution",1000);


%% Functions

function IR_perim = load_IR(IR_dir,IR_perim_type,polyshape_true)    % load IR perimeter as polyshape or geoshape objects for perimeters with or without projection
    if IR_perim_type == 0    % perimeter with projection
        IR_perimeter = readgeotable(IR_dir);
        p1 = IR_perimeter.Shape.ProjectedCRS;
        IR_perimeter_2 = geotable2table(IR_perimeter,["x1","y1"]);
        x1 = IR_perimeter_2.x1{1};
        y1 = IR_perimeter_2.y1{1};
        [lat,lon] = projinv(p1,x1,y1);
        if polyshape_true == 1
            IR_perim = polyshape(lon,lat);
        elseif polyshape_true == 0
            IR_perim = geoshape(lat,lon);
        end
    elseif IR_perim_type == 1    % perimeter without projection
        IR_perimeter = readgeotable(IR_dir);
        IR_perimeter_2 = geotable2table(IR_perimeter,["Lat","Lon"]);
        lat = IR_perimeter_2.Lat{1};
        lon = IR_perimeter_2.Lon{1};
        if polyshape_true == 1
            IR_perim = polyshape(lon,lat);
        elseif polyshape_true == 0
            IR_perim = geoshape(lat,lon);
        end
    end
end

function [high_conf_meas,high_nom_conf_meas] = load_measurement_images(high_conf_meas_dir,high_nom_conf_meas_dir)    % load VIIRS measurement images
    high_conf_meas = load(high_conf_meas_dir).measurement;
    high_nom_conf_meas = load(high_nom_conf_meas_dir).measurement;
end

function [lat_SVM,lon_SVM,tign_SVM] = load_SVM(SVM_dir)    % load SVM fire arrival time predictions
    lat_SVM = double(ncread(SVM_dir,'lat'));
    lon_SVM = double(ncread(SVM_dir,'lon'));
    tign_SVM = double(ncread(SVM_dir,'tign'));
end

function [high_conf_samples,high_nom_conf_samples,weighted_mean,weighted_SD,weighted_mean_plus_SD,weighted_mean_minus_SD] = load_GAN_samples_mixture_model(high_conf_samples_dir, high_nom_conf_samples_dir, weight)    % load cWGAM predictions and combine them in weighted fashion
    % load samples
    high_conf_samples = load(high_conf_samples_dir).samples;
    high_nom_conf_samples = load(high_nom_conf_samples_dir).samples;

    % Sample weighting
    high_conf_weight = weight;
    high_nom_conf_weight = 1 - high_conf_weight;

    % Statistics
    high_conf_mean = squeeze(mean(high_conf_samples));
    high_conf_var = squeeze(std(high_conf_samples)).^2;
    high_nom_conf_mean = squeeze(mean(high_nom_conf_samples));
    high_nom_conf_var = squeeze(std(high_nom_conf_samples)).^2;

    % Weight and combine statistics from difference measurement confidence levels
    weighted_mean = high_conf_weight*high_conf_mean + high_nom_conf_weight*high_nom_conf_mean;
    weighted_var = high_conf_weight*high_conf_var + high_nom_conf_weight*high_nom_conf_var + high_conf_weight*high_nom_conf_weight*(high_conf_mean-high_nom_conf_mean).^2;
    weighted_SD = weighted_var.^(1/2);

    weighted_mean_plus_SD = weighted_mean + weighted_SD;
    weighted_mean_minus_SD = weighted_mean - weighted_SD;
end

function [lat,lon,latlim,lonlim] = GAN_geolocation(center_lat,center_lon)    % determine geolocation for cWGAN predictions (determined in same fashion as used when constructing measurement image inputs for cWGAN)
    x = 0:60:60*512;
    y = x;
    z = zeros(length(x));
    bottom_left_lat = center_lat - 0.12;
    bottom_left_lon = center_lon - 0.12;
    origin = [bottom_left_lat,bottom_left_lon,0];
    [lat_nodes,lon_nodes] = local2latlon(x,y,z,origin);
    lat_nodes = rot90(lat_nodes);

    lat = zeros(512,512);
    lon = zeros(512,512);
    for k = 1:length(lat(1,:))  
        for kk = 1:length(lat(1,:))
            lat(k,kk) = (lat_nodes(k,kk) + lat_nodes(k,kk+1) + lat_nodes(k+1,kk) + lat_nodes(k+1,kk+1))/4;    % create matricies of lats and lons 
            lon(k,kk) = (lon_nodes(k,kk) + lon_nodes(k,kk+1) + lon_nodes(k+1,kk) + lon_nodes(k+1,kk+1))/4;
        end
    end

    latlim = [min(lat,[],'all'),max(lat,[],'all')];    % get lat and lon limits
    lonlim = [min(lon,[],'all'),max(lon,[],'all')];
end

function make_measurement_plot_for_figure(measurement,IR_perim,lat,lon,latlim,lonlim,font_size,IR_option,cb_on)    % make plots of VIIRS measurement image
    ax = usamap(latlim,lonlim);
    geoshow(lat,lon,measurement*72,DisplayType="texturemap"); 
    colormap(hot);
    ax.CLim = [12 72];
    if cb_on == 1
        cb = colorbar;
        cb.Ticks = 0:12:72;
        cb.FontSize = font_size;
    end
    if IR_option == 1
        geoshow(IR_perim); 
    end
    setm(gca,'FontSize',font_size);
end

function make_prediction_statistics_plot_for_fig(statistic,cmap,clim_on,lat,lon,latlim,lonlim,font_size,cb)    % make plot of predictions from cWGAN and SVM methods
    ax = usamap(latlim,lonlim);
    geoshow(lat,lon,statistic*72,DisplayType="texturemap"); 
    colormap(ax,cmap);
    if cb == 1
        cb = colorbar;
        cb.FontSize = font_size;
        if clim_on == 1
            cb.Ticks = 0:12:72;
        elseif clim_on == 0
            cb.Ticks = 0:6:24;
        end
    end
    if clim_on == 1
        ax.CLim = [12 72];
%         cb.Label.String = 'Hours since start of ignition day';
    else
        ax.CLim = [0 24];
%         cb.Label.String = 'Hours';
    end
%     cb.Label.FontSize = font_size;
    setm(gca,'FontSize',font_size);
end

function polygon = create_poly_from_raster(lat,lon,raster,perimeter_time)    % create polygon out of predictions and corresponding geolocation for SC computation
    M = contourm(lat,lon,raster,LevelList=perimeter_time); 

    for k = 1:length(M(1,:))
        if M(1,k) == perimeter_time
            M(:,k) = NaN;
        end
    end    
    M = M(:,2:end);

    polygon = polyshape(M(1,:),M(2,:));

    ispolycw(M(2,:),M(1,:));
end

function [SC,POD,FAR,A,B,C] = area_discrepancy_calculations(IR_perim,predicted_perim)    % compute SC, POD, FAR
    A = intersect(predicted_perim,IR_perim);
    B = subtract(IR_perim,predicted_perim);
    C = subtract(predicted_perim,IR_perim);
    A_area = area(A);
    B_area = area(B);
    C_area = area(C);
    SC = 2*A_area/(2*A_area + B_area + C_area);
    POD = A_area/(A_area + B_area);
    FAR = C_area/(A_area + C_area);
end

function make_area_discrepancy_subplots(A,B,C,plot_title,lat,lon,latlim,lonlim,weighted_mean,font_size)    % make plot showing area discrepancies between predicitons and IR perimeters
    geoshow(lat,lon,weighted_mean*72,'DisplayType','contour','LevelList',-100); hold on;
    plot(A,FaceColor='black'); 
    plot(B,FaceColor='blue');
    plot(C,FaceColor='red');
    axis([lonlim latlim]);
    set(gca,'xtick',ceil(lonlim(1)*10)/10:0.1:floor(lonlim(2)*10)/10);
    set(gca,'ytick',ceil(latlim(1)*10)/10:0.1:floor(latlim(2)*10)/10);
    grid on;
    xlabel('lon',FontSize=font_size);
    ylabel('lat',FontSize=font_size);
%     title(plot_title,FontSize=font_size);
    set(gca,'FontSize',font_size);
    legend('A','B','C');
end
