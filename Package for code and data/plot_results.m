clear all;
close all;
clc;
% results in simulated single-pixel imaging with MNIST database using Bernoulli-distributed, Laplacian-distributed and Gaussian-distributed likelihood functions
% compression ratio is 16x; 25dB of noise is added to the simulated measurement data
% Matlab R2020a

%%
% load the ground truth images in the testing dataset
load('ground_truth.mat','x');
object = x(901:end,:,:);
object = permute(object,[2 3 1]);

% load the input images to BCNN in the testing dataset
load('network_input.mat','Y_image');
Image_input_16X = Y_image(901:end,:,:);
Image_input_16X = permute(Image_input_16X,[2 3 1]);



%% results using Bernoulli-distributed likelihood function

% load the predicted images in the testing dataset after applying the Monte Carlo Dropout
load('bernoulli\25dB\Image_MCDropout.mat','Image_MCDropout');
error_temp = double(Image_MCDropout);
Image_16X = mean(error_temp(:,:,:,1,:),5);
Image_16X = permute(Image_16X,[2 3 1]);

% calculate the data and model uncertainty
Image_16X_error_data = sqrt(mean((error_temp(:,:,:,1,:).*(1-error_temp(:,:,:,1,:))),5));
Image_16X_error_data = permute(Image_16X_error_data,[2 3 1]);
Image_16X_error_model = std(error_temp(:,:,:,1,:),1,5);
Image_16X_error_model = permute(Image_16X_error_model,[2 3 1]);


for i = 1:100
    loss1_inverse_16X(i) = mean2(abs(Image_16X(:,:,i)-object(:,:,i))); % MAE
    loss2_inverse_16X(i) = (1-ssim(Image_16X(:,:,i),object(:,:,i)))/2; % DSSIM
    % correlation coefficient between the true absolute error and the predicted uncertainty
    corr_coef_temp = corrcoef(abs(object(:,:,i) - Image_16X(:,:,i)),sqrt(Image_16X_error_model(:,:,i).^2 + Image_16X_error_data(:,:,i).^2)); 
    corr_coef_16X(i) = corr_coef_temp(1,2);
end
% mean and std of MAE, SSIM and correlation coefficient R
MAE_avg_bernoulli = mean(loss1_inverse_16X);
SSIM_avg_bernoulli = mean(1-2*loss2_inverse_16X);
R_avg_bernoulli = mean(corr_coef_16X);

MAE_std_bernoulli = std(loss1_inverse_16X);
SSIM_std_bernoulli = std(1-2*loss2_inverse_16X);
R_std_bernoulli = std(corr_coef_16X);

%% results using laplacian-distributed likelihood function

% load the predicted images in the testing dataset after applying the Monte Carlo Dropout
load('laplacian\25dB\Image_MCDropout.mat','Image_MCDropout');
error_temp = double(Image_MCDropout);
Image_16X_lap = mean(error_temp(:,:,:,1,:),5);
Image_16X_lap = permute(Image_16X_lap,[2 3 1]);

% calculate the data and model uncertainty
Image_16X_lap_error_data = sqrt(mean(2.*(error_temp(:,:,:,2,:).^2),5));
Image_16X_lap_error_data = permute(Image_16X_lap_error_data,[2 3 1]);
Image_16X_lap_error_model = std(error_temp(:,:,:,1,:),1,5);
Image_16X_lap_error_model = permute(Image_16X_lap_error_model,[2 3 1]);


for i = 1:100
    loss1_inverse_16X_lap(i) = mean2(abs(Image_16X_lap(:,:,i)-object(:,:,i))); % MAE
    loss2_inverse_16X_lap(i) = (1-ssim(Image_16X_lap(:,:,i),object(:,:,i)))/2; % DSSIM
    % correlation coefficient between the true absolute error and the predicted uncertainty
    corr_coef_temp = corrcoef(abs(object(:,:,i) - Image_16X_lap(:,:,i)),sqrt(Image_16X_lap_error_model(:,:,i).^2 + Image_16X_lap_error_data(:,:,i).^2));
    corr_coef_16X_lap(i) = corr_coef_temp(1,2);
end

% mean and std of MAE, SSIM and correlation coefficient
MAE_avg_laplacian = mean(loss1_inverse_16X_lap);
SSIM_avg_laplacian = mean(1-2*loss2_inverse_16X_lap);
R_avg_laplacian = mean(corr_coef_16X_lap);

MAE_std_laplacian = std(loss1_inverse_16X_lap);
SSIM_std_laplacian = std(1-2*loss2_inverse_16X_lap);
R_std_laplacian = std(corr_coef_16X_lap);

%% results using gaussian-distributed likelihood function

% load the predicted images in the testing dataset after applying the Monte Carlo Dropout
load('gaussian\25dB\Image_MCDropout.mat','Image_MCDropout');
error_temp = double(Image_MCDropout);
Image_16X_gauss = mean(error_temp(:,:,:,1,:),5);
Image_16X_gauss = permute(Image_16X_gauss,[2 3 1]);

% calculate the data and model uncertainty
Image_16X_gauss_error_data = sqrt(mean((error_temp(:,:,:,2,:).^2),5));
Image_16X_gauss_error_data = permute(Image_16X_gauss_error_data,[2 3 1]);
Image_16X_gauss_error_model = std(error_temp(:,:,:,1,:),1,5);
Image_16X_gauss_error_model = permute(Image_16X_gauss_error_model,[2 3 1]);


for i = 1:100
    loss1_inverse_16X_gauss(i) = mean2(abs(Image_16X_gauss(:,:,i)-object(:,:,i))); % MAE
    loss2_inverse_16X_gauss(i) = (1-ssim(Image_16X_gauss(:,:,i),object(:,:,i)))/2; % DSSIM
    % correlation coefficient between the true absolute error and the predicted uncertainty
    corr_coef_temp = corrcoef(abs(object(:,:,i) - Image_16X_gauss(:,:,i)),sqrt(Image_16X_gauss_error_model(:,:,i).^2 + Image_16X_gauss_error_data(:,:,i).^2));
    corr_coef_16X_gauss(i) = corr_coef_temp(1,2);
end

% mean and std of MAE, SSIM and correlation coefficient
MAE_avg_gaussian = mean(loss1_inverse_16X_gauss);
SSIM_avg_gaussian = mean(1-2*loss2_inverse_16X_gauss);
R_avg_gaussian = mean(corr_coef_16X_gauss);

MAE_std_gaussian = std(loss1_inverse_16X_gauss);
SSIM_std_gaussian = std(1-2*loss2_inverse_16X_gauss);
R_std_gaussian = std(corr_coef_16X_gauss);

%% plots for the Bernoulli case

% ground truth image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(object(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% input image to BCNN
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_input_16X(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% predicted image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% true absolute error
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(abs(object(:,:,3) - Image_16X(:,:,3))); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(sqrt(Image_16X_error_model(:,:,3).^2 + Image_16X_error_data(:,:,3).^2)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted data uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_error_data(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted model uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_error_model(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);

%% plots for the Laplacian case

% ground truth image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(object(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% input image to BCNN
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_input_16X(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% predicted image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_lap(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% true absolute error
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(abs(object(:,:,3) - Image_16X_lap(:,:,3))); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(sqrt(Image_16X_lap_error_model(:,:,3).^2 + Image_16X_lap_error_data(:,:,3).^2)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted data uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_lap_error_data(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted model uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_lap_error_model(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);

%% plots for the Gaussian case

% ground truth image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(object(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% input image to BCNN
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_input_16X(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% predicted image
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_gauss(:,:,3)); colormap gray;axis equal;axis off;caxis([0 1]);

% true absolute error
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(abs(object(:,:,3) - Image_16X_gauss(:,:,3))); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(sqrt(Image_16X_gauss_error_model(:,:,3).^2 + Image_16X_gauss_error_data(:,:,3).^2)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted data uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_gauss_error_data(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);

% predicted model uncertainty
fh = figure; 
ah = axes('Units','Normalize','Position',[0 0 1 1]);
fh.Position(3) = fh.Position(4);
imagesc(Image_16X_gauss_error_model(:,:,3)); colormap jet;axis equal;axis off;caxis([0 1]);
