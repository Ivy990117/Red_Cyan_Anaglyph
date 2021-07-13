clc
clear all 
%% mannually selecting key points
I1 = imread('FD_Object/HG_2.jpg');
I2 = imread('FD_Object/HG_1.jpg');
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

load('p3_point.mat');
load('p4_point.mat');

%show mannualy selected matching points 
% figure(2);
% showMatchedFeatures(I1,I2,mannual_p4,mannual_p3,'montage');
% title('Mannual matched points');

%% automatically selecting keypoints
% ROI_left = [1485 2046 537 770];
% ROI_right = [1747 1769 503 718];
ROI_left = [1743 1759 (2184 - 1743) (2548 - 1759)]; %set the region
%ROI_right = [1740 1586 (2668 - 1740) (2388 - 1586)];
% left_points = detectSURFFeatures(image1,'ROI',ROI_left);
% right_points = detectSURFFeatures(image2,'ROI',ROI_right);

%detect keypoints by SURF method
left_points = detectSURFFeatures(image1,'ROI',ROI_left);
right_points = detectSURFFeatures(image2);

%show strongest keypoints
% figure(3); 
% subplot(121);
% imshow(image1);
% title('100 Strongest Feature Points from Left Image');
% hold on;
% plot(selectStrongest(left_points, 100));
% subplot(122);
% imshow(image2);
% title('100 Strongest Feature Points from Middle Image');
% hold on;
% plot(selectStrongest(right_points, 100));
% hold off

%matching keypoints
[features_left,valid_points_left] = extractFeatures(image1,left_points);
[features_right,valid_points_right] = extractFeatures(image2,right_points);
index_pairs = matchFeatures(features_left, features_right);
matched_points_left = valid_points_left(index_pairs(:,1),:);
matched_points_right = valid_points_right(index_pairs(:,2),:);

%show automatically selected matching keypoints
% figure(3);
% showMatchedFeatures(image1,image2,matched_points_left,matched_points_right,'montage');
% legend('matched points - left','matched points - right');
% title('Automatic matched points');

[tform, inlier_points_left,inlier_points_right] = estimateGeometricTransform(matched_points_left, matched_points_right, 'projective');


% inlier_points_left   = matched_points_left(inlier1, :);
% inlier_points_right = matched_points_right(inlier2, :);

%show inlier keypoints
figure(2);
showMatchedFeatures(I1,I2,inlier_points_left,inlier_points_right,'montage');
title('Automatic matched points');

auto_p3 = double(inlier_points_left.Location);
auto_p4 = double(inlier_points_right.Location);
%% compare the mean difference of two methods
window_size = 25;
l = floor(window_size/2);
[neighbourhood_auto1,neighbourhood_auto2] = WindowGeneration(auto_p3, auto_p4,l,image1,image2);
[neighbourhood_manu1,neighbourhood_manu2] = WindowGeneration(mannual_p3, mannual_p4,l,image1,image2);

for i=1:size(neighbourhood_auto1, 1)
    for j=1:size(neighbourhood_auto2, 1)
        dist_auto(i,j) = sqrt(sum((neighbourhood_auto1{i}'- neighbourhood_auto2{j}').^2));
    end
end
for i=1:size(neighbourhood_manu1, 1)
    for j=1:size(neighbourhood_manu2, 1)
        dist_manu(i,j) = sqrt(sum((neighbourhood_manu1{i}' - neighbourhood_manu2{j}').^2));
    end
end 

%mean difference by auto method
diff_values = sort(dist_auto(:));
mean_diff_auto = mean(diff_values(1:10));
s1 = ['Mean difference using auto corresponding method is',num2str(mean_diff_auto),'.'];
disp(s1);

%mean difference by mannual method
mean_diff_manu = mean(dist_manu(:));
s2 = ['Mean difference using manual corresponding method is',num2str(mean_diff_manu),'.'];
disp(s2);
%% calculate the homography matrix
%construct matrix of keypoints by auto method
for index=1:size(auto_p3, 1)
    index1 = auto_p3(index,:);
    index2 = auto_p4(index,:);
    D1(:,index) = [index1(1),index1(2)];
    D2(:,index) = [index2(1),index2(2)];
end

% construct matrix of keypoints by mannual method
% for i=1:10
%     index1 = mannual_p3(i,:);
%     index2 = mannual_p4(i,:);
%     D1(:,i) = [index1(1);index1(2)];
%     D2(:,i) = [index2(1);index2(2)];
% end

%Plot the Putative matches
threshold = 0.1;
num = 15;
iternations = 1000;

% compute Homography matrix H
[H, inliers, point1, point2, residue, MSE] = fHmatrix(D1, D2, threshold,  num, iternations);
%[H, inliers, point1, point2, residue, MSE] = fHmatrix(D1, D2, threshold,  num, iternations);
plot_matches(point1, point2, I1, I2);
title('Inliner Keypoint Correspondences in use');

%%
s3 = ['Mean squared error using auto corresponding method is',num2str(MSE),'.'];
disp(s3);