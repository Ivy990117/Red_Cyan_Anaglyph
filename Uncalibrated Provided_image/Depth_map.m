%% Read the rectified images
I1 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Uncalibrated Provided_image/I1Rect_Yellow.png');
I2 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Uncalibrated Provided_image/I2Rect_Yellow.png');

%% Convert the rectified input color images to grayscale images.
J1 = rgb2gray(I1);
J2 = rgb2gray(I2);

%% Compute the disparity map
disparityRange = [0 64];
%disparityMap = disparitySGM(J1,J2);
disparityMap = disparity(J1,J2,'DisparityRange',disparityRange,'UniquenessThreshold',15);

%% Display disparity map
figure;
imshow(disparityMap,disparityRange);
%imshow('Disparity_Map.png');
title('Disparity Map');
colormap jet
colorbar