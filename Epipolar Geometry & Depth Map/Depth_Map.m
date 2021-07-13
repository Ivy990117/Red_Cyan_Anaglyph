
%% Read the rectified images
% I1 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Epipolar Geometry & Depth Map/I1Rect_Yellow.png');
% I2 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Epipolar Geometry & Depth Map/I2Rect_Yellow.png');

I1 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Epipolar Geometry & Depth Map/I1r.jpg');
I2 = imread('/Users/liujiaoyang/Documents/MATLAB/Three_Dimensional_Reconstruction/Epipolar Geometry & Depth Map/I2r.jpg');

%% Convert the rectified input color images to grayscale images.
J1 = rgb2gray(I1);
J2 = rgb2gray(I2);

%% Compute the disparity map
disparityRange = [0 64];
%disparityMap = disparitySGM(J1,J2);
disparityMap = disparity(J1,J2,'DisparityRange',disparityRange,'UniquenessThreshold',20);

%% Display disparity map
figure;
imshow(disparityMap,disparityRange,'InitialMagnification',50);
%imshow('Disparity_Map.png');
title('Disparity Map');
% colormap jet
colorbar