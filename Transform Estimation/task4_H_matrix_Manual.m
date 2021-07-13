clc
clear all 
%% mannually selecting key points
I1 = imread('FD_Object/HG_2.jpg');
I2 = imread('FD_Object/HG_1.jpg');
%cpselect(I1,I2);

image1 = im2double(I1);
image2 = im2double(I2);
image1 = rgb2gray(image1);
image2 = rgb2gray(image2);

%show original images
figure(1);
subplot(1,2,1);
imshow(I1);
title('Original left images');
subplot(1,2,2);
imshow(I2);
title('Original right images');
% load('p1_point.mat');
% load('p2_point.mat');
load('p3_point.mat');
load('p4_point.mat');
% mannual_left=zeros(35,2);
% mannual_left(1:25,:)=p1_point;
% mannual_left(26:35,:)=mannual_p4;
% mannual_right=zeros(35,2);
% mannual_right(1:25,:)=p2_point;
% mannual_right(26:35,:)=mannual_p3;
%show mannualy selected matching points 

figure(2);
showMatchedFeatures(I1,I2,mannual_p4,mannual_p3,'montage');
title('Mannual matched points');
%showMatchedFeatures(I1,I2,mannual_left,mannual_right,'montage');

%%
window_size = 25;
l = floor(window_size/2);
[neighbourhood_manu1,neighbourhood_manu2] = WindowGeneration(mannual_p3,mannual_p4,l,image1,image2);
%[neighbourhood_manu1,neighbourhood_manu2] = WindowGeneration(mannual_left,mannual_right,l,image1,image2);

for i=1:size(neighbourhood_manu1, 1)
    for j=1:size(neighbourhood_manu2, 1)
        dist_manu(i,j) = sqrt(sum((neighbourhood_manu1{i}' - neighbourhood_manu2{j}').^2));
    end
end 


%mean difference by mannual method
mean_diff_manu = mean(dist_manu(:));
s2 = ['Mean difference using manual corresponding method is',num2str(mean_diff_manu),'.'];
disp(s2);

%% construct matrix of keypoints by mannual method
for i=1:10
    index1 = mannual_p4(i,:);
    index2 = mannual_p3(i,:);
%     index1 = mannual_left(i,:);
%     index2 = mannual_right(i,:);
    D1(:,i) = [index1(1);index1(2)];
    D2(:,i) = [index2(1);index2(2)];
end

%% Plot the Putative matches
threshold = 0.1;
num = 5;
iternations = 1000;

% compute Homography matrix H
%[H, inliers, point1, point2, residue, MSE] = fHmatrix(D1, D2, threshold,  num, iternations);
[H, inliers, point1, point2, residue, MSE] = fHmatrix(D1, D2, threshold,  num, iternations);
plot_matches(point1, point2, I1, I2);
title('Manual keypoint correspondences in use');

%%
s3 = ['Mean squared error using mannual method is',num2str(MSE),'.'];
disp(s3);