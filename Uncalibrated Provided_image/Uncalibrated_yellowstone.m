%% Step 1: Read Stereo Image Pair
% Read in two color images of the same scene, which were taken from
% different positions. Then, convert them to grayscale. Colors are not
% required for the matching process.
I1 = imread('yellowstone_left.png');
I2 = imread('yellowstone_right.png');

% Convert to grayscale.
I1gray = rgb2gray(I1);
I2gray = rgb2gray(I2);

%%
% Display both images side by side. Then, display a color composite
% demonstrating the pixel-wise differences between the images.
figure;
imshowpair(I1, I2,'montage');
title('I1 (left); I2 (right)');
figure; 
imshow(stereoAnaglyph(I1,I2));
title('Composite Image (Red - Left Image, Cyan - Right Image)');

%%
% There is an obvious offset between the images in orientation and
% position. The goal of rectification is to transform the images, aligning
% them such that corresponding points will appear on the same rows in both
% images.

%% Step 2: Collect Interest Points from Each Image

blobs1 = detectSURFFeatures(I1gray, 'MetricThreshold', 2000);
blobs2 = detectSURFFeatures(I2gray, 'MetricThreshold', 2000);

%%
% Visualize the location and scale of the thirty strongest SURF features in
% I1 and I2.  
figure; 
imshow(I1);
hold on;
plot(selectStrongest(blobs1, 30));
title('Thirty strongest SURF features in I1');

figure; 
imshow(I2); 
hold on;
plot(selectStrongest(blobs2, 30));
title('Thirty strongest SURF features in I2');

%% Step 3: Find Putative Point Correspondences
% Use the |extractFeatures| and |matchFeatures| functions to find putative
% point correspondences. For each blob, compute the SURF feature vectors
% (descriptors).
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

%%
% Show matching points on top of the composite image, which combines stereo
% images. Notice that most of the matches are correct, but there are still
% some outliers.
figure; 
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
legend('Putatively matched points in I1', 'Putatively matched points in I2');

%% Step 4: Remove Outliers Using Epipolar Constraint
% The correctly matched points must satisfy epipolar constraints. 
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

figure;
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
legend('Inlier points in I1', 'Inlier points in I2');

%% Step 5: Rectify Images
% Compute the rectification transformations. These can be used to transform the images,
% such that the corresponding points will appear on the same rows.
[t1, t2] = estimateUncalibratedRectification(fMatrix, ...
  inlierPoints1.Location, inlierPoints2.Location, size(I2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);

%%
% Rectify the stereo images, and display them as a stereo anaglyph.
% You can use red-cyan stereo glasses to see the 3D effect.
[I1Rect, I2Rect] = rectifyStereoImages(I1, I2, tform1, tform2);
figure;
imshow(stereoAnaglyph(I1Rect, I2Rect));
title('Rectified Stereo Images (Red - Left Image, Cyan - Right Image)');

