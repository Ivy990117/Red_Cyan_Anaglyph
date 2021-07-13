clc
close all
clear all

% Reading the stereo pair of scene
image1=imread('FD_object/FG_1.jpg');
image2=imread('FD_object/FG_2.jpg');

left = image1;
right = image2;

% Load the left points and right points
load p5_point.mat
load p6_point.mat

load el_left;
load el_right;

left_points = auto_p3;
right_points = auto_p4;

% Plotting the point selected
figure(1);
subplot(121);
imshow(left),title('Points on the left image'),title('Left Image and 15 points')
hold on
plot(left_points(:,1),left_points(:,2),'b*')
plot(el_left(:,1),el_left(:,2),'r*')


subplot(122);
imshow(right),title('Points on the right image'),title('Right Image and 15 corresponding points')
hold on
plot(right_points(:,1),right_points(:,2),'b*',el_right(:,1),el_right(:,2),'r*')


%% Calculate the F-matrix
% Normalize the points( Hartley preconditioning algorithm)
% The coordinates of corresponding points can have a wide range leading to numerical instabilities.
% It is better to first normalize them so they have average 0 and stddev 1 (mean distance from the center is sqrt(2)
% and denormalize F at the end

l=left_points';
r=right_points';
% Normalising left points
centl=mean(l, 2); %x_bar and y_bar
tl=bsxfun(@minus,l, centl);

% compute the scale to make mean distance from centroid sqrt(2)
meanl=mean(sqrt(sum(tl.^2)));
if meanl>0 % protect against division by 0
    sl=sqrt(2)/meanl;
else
    sl=1;
end
T=diag(ones(1,3)*sl);
T(1:end-1,end)=-sl*centl;
T(end)=1;
if size(l,1)>2
    left_normPoints=T*l;
else
    left_normPoints=tl*sl;
end
% Normalising the right points
centr=mean(r, 2); %x_bar and y_bar
tr=bsxfun(@minus,r,centr);

% compute the scale to make mean distance from centroid sqrt(2)
meanr=mean(sqrt(sum(tr.^2)));
if meanr>0 % protect against division by 0
    sr=sqrt(2)/meanr;
else
    scaler=1;
end
T_bar=diag(ones(1,3)*sr);
T_bar(1:end-1,end)=-sr*centr;
T_bar(end)=1;
if size(r,1)>2
    right_normPoints=T_bar*r ;
else
    right_normPoints=tr*sr;
end
%% Expressing the linear equation
u=left_normPoints(1,:)';
v=left_normPoints(2,:)';
ud=right_normPoints(1,:)';
vd=right_normPoints(2,:)';

% Defining the image points in one matrix
P=[u.*ud,v.*ud,ud,u.*vd,v.*vd,vd,u,v,ones(size(u,1),1)]; 
% % Perform SVD of P
[U,S,V]=svd(P);
[min_val,min_index]=min(diag(S(1:9,1:9)));
% % m is given by right singular vector of min. singular value
m=V(1:9,min_index);
% Projection matrix reshaping
F=[m(1:3,1)';m(4:6,1)';m(7:9,1)'];
% F1=F;
% To enforce rank 2 constraint:
% Find the SVD of F: F = Uf.Df.VfT
% Set smallest s.v. of F to 0 to create D?f
% Recompute F: F = Uf.D?f.VfT
[Uf,Sf,Vf]=svd(F);
Sf(end)=0;
F=Uf*Sf*Vf';
% Denormalise F and transform back to original scale
F=T_bar'*F*T;
% % Normalize the fundamental matrix.
F=F/norm(F);
if F(end)<0
  F=-F;
end

%%
[t1, t2] = estimateUncalibratedRectification(F, ...
  left_points, right_points, size(image2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);

[I1Rect, I2Rect] = rectifyStereoImages(image1, image2, tform1, tform2);

%%
figure;
imshowpair(I1Rect, I2Rect,'montage');