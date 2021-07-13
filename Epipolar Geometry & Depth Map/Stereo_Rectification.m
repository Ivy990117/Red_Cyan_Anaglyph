close all;
clear all;
clc;


%% Step 1: Read Stereo Image Pair
I1 = imread('FD_left.jpg');
I2 = imread('FD_right.jpg');

%cpselect(I1,I2);
% Convert to grayscale.
I1gray = rgb2gray(I1);
I2gray = rgb2gray(I2);

%%
% Display both images side by side. Then, display a color composite
% demonstrating the pixel-wise differences between the images.
figure(1);
subplot(121);
imshow(I1);
title('Left image');
subplot(122);
imshow(I2);
title('Right image');

%% Step 2: Collect KeyPoints from Each Image
% The rectification process requires a set of point correspondences between
% the two images. Use |detectSURFFeatures| to find blob-like features in both
% images.
blobs1 = detectSURFFeatures(I1gray, 'MetricThreshold', 2000);
blobs2 = detectSURFFeatures(I2gray, 'MetricThreshold', 2000);

%%
% Visualize the location and scale of the 100 strongest SURF features in
% I1 and I2.  
figure(2);
subplot(121);
imshow(I1);
hold on;
plot(selectStrongest(blobs1, 100));
title('100 strongest SURF features in left image I1');
subplot(122);
imshow(I2); 
hold on;
plot(selectStrongest(blobs2,100));
title('100 strongest SURF features in right image I2');

%% Step 3: Find Putative Point Correspondences

[features1, validBlobs1] = extractFeatures(I1gray, blobs1);
[features2, validBlobs2] = extractFeatures(I2gray, blobs2);

%%
% Use the sum of absolute differences (SAD) metric to determine indices of
% matching features.
indexPairs = matchFeatures(features1, features2, 'Metric', 'SAD', ...
  'MatchThreshold', 5);

%%
% Retrieve locations of matched points for each image.
matchedPoints1 = validBlobs1(indexPairs(:,1),:);
matchedPoints2 = validBlobs2(indexPairs(:,2),:);


figure(4);
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2,'montage');
legend('Matched points in I1', 'Matched points in I2');

%% Step 4: Remove Outliers Using Epipolar Constraint

[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
  matchedPoints1, matchedPoints2, 'Method', 'RANSAC', ...
  'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);
  
if status ~= 0 || isEpipoleInImage(fMatrix, size(I1)) ...      
  || isEpipoleInImage(fMatrix', size(I2))
  error(['Either not enough matching points were found or '...
         'the epipoles are inside the images. You may need to '...
         'inspect and improve the quality of detected features ',...
         'and/or improve the quality of your images.']);
end

inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

figure(5);
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2,'montage');
legend('Inlier points in I1', 'Inlier points in I2');


%% Step 5: Rectify Images

[H1,H2, K] = rectifyF(inlierPoints1.Location, inlierPoints2.Location, [size(I2,2),size(I2,1)] );

mlx = htx(H1,inlierPoints1.Location'); 
mrx = htx(H2,inlierPoints2.Location');
 % Sampson error wrt to F=skew([1 0 0])
err = sqrt(sum(F_sampson(skew([1 0 0]),mlx,mrx).^2)/(length(mlx)-1));
fprintf('Rectification Sampson RMSE: %0.5g pixel \n',err);
         
% projective MPP (Euclidean only if K is guessed right)
Pn1=[K,[0;0;0]]; Pn2=[K,[1;0;0]];
[I1r,I2r, bb1, bb2] = imrectify(I1,I2,H1,H2,'valid');
    
    % xshift =  bb1(1)  - bb2(1)
    
Pn1 = [1 0 -bb1(1);  0 1 -bb1(2); 0 0 1] *Pn1;
Pn2 = [1 0 -bb2(1);  0 1 -bb2(2); 0 0 1] *Pn2;


%% plot result
points1=load('epipolarline_points1');
points2=load('epipolarline_points2');
points_left = p2t(H1,points1);
points_right = p2t(H2,points2);

figure;

subplot(2,2,1)
image(I1);
axis image
hold on
title('Original left image');
x2 = size(I1,2);
for i =1:size(points1,2)
    plot (points1(1,i), points1(2,i),'r+','MarkerSize',12);
end
hold on;


for i =1:size(points2,2)
  epiLines = epipolarLine(fMatrix',points2(:,i)');
  temp = lineToBorderPoints(epiLines,size(I1));
  line(temp(:,[1,3])',temp(:,[2,4])','Color','b');
end

subplot(2,2,2)
image(I2);
axis image
hold on
title('Original right image')

for i =1:size(points1,2)
  epiLines = epipolarLine(fMatrix,points1(:,i)');
  temp = lineToBorderPoints(epiLines,size(I1));
  line(temp(:,[1,3])',temp(:,[2,4])','Color','r');
end
hold on;

for i =1:size(points2,2)
    plot (points2(1,i), points2(2,i),'b+','MarkerSize',12);
end

subplot(2,2,3)
image(uint8(I1r));
axis image
hold on
title('Rectified left image');
x2 = size(I1r,2);
for i =1:size(points_left,2)
    plot (points_left(1,i)-bb1(1), points_left(2,i)-bb1(2),'r+','MarkerSize',12);
end
hold on;

x1=0;
x2 = size(I1r,2);
for i =1:size(points_right,2)
    liner = star([1 0 0])  * [points_right(:,i) - bb1(1:2) ;  1];
    [y1,y2]=plotseg(liner,x1,x2);
    plot([x1 x2], [y1, y2],'b');
end


subplot(2,2,4)
image(uint8(I2r));
axis image
hold on
title('Rectified right image')
x1=0;
x2 = size(I2r,2);
for i =1:size(points_left,2)
    liner = star([1 0 0])  * [points_left(:,i) - bb2(1:2) ;  1];
    [y1,y2]=plotseg(liner,x1,x2);
    plot([x1 x2], [y1, y2],'r');
end
hold on;

for i =1:size(points_right,2)
    plot (points_right(1,i)-bb2(1), points_right(2,i)-bb2(2),'b+','MarkerSize',12);
end
hold off

%%show Red-Cyan Anaglyph
rc_Anaglyph = stereoAnaglyph(uint8(I1r), uint8(I2r));
figure;
imshow(rc_Anaglyph);
title('Red-Cyan Anaglypn');