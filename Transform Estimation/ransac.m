%% compute Homography matrix H (from D2 to D1)
function[H, num_inlier, point1, point2, in_res] = ransac(D1, D2, threshold, num, iternations)
    count = 0;
    D1(3,:) = 1;
    D2(3,:) = 1;
    n = size(D1, 2); %the number of matched keypoints provided
    best_n_inlier = 0; %initialize the optimal number of keypoints to construct H
    best_H = [];
    
    for i=1:iternations
        s = randsample(n, num); %returns num=4 values sampled uniformly at random, from the integers 1 to n.
        %randomly select num=4 points cooresponding to s
        x = D1(1, s);
        y = D1(2, s); 
        xp = D2(1, s);
        yp = D2(2, s);
        
        %construct known matrix P (P*h=0)
        P=zeros(2*num,9);
        for irow=1:num
            %odd rows
            P(2*irow-1,:)=[x(irow), y(irow) 1 0 0 0 -x(irow)*xp(irow) -y(irow)*xp(irow) -xp(irow)];
            %even rows
            P(2*irow,:)=[0 0 0 x(irow) y(irow) 1 -x(irow)*yp(irow) -y(irow)*yp(irow) -yp(irow)];
        end
        
%         P = [
%             x(1) y(1) 1 0 0 0 -x(1)*xp(1) -y(1)*xp(1) -xp(1);
%             0 0 0 x(1) y(1) 1 -x(1)*yp(1) -y(1)*yp(1) -yp(1);
%             x(2) y(2) 1 0 0 0 -x(2)*xp(2) -y(2)*xp(2) -xp(2);
%             0 0 0 x(2) y(2) 1 -x(2)*yp(2) -y(2)*yp(2) -yp(2);
%             x(3) y(3) 1 0 0 0 -x(3)*xp(3) -y(3)*xp(3) -xp(3);
%             0 0 0 x(3) y(3) 1 -x(3)*yp(3) -y(3)*yp(3) -yp(3);
%             x(4) y(4) 1 0 0 0 -x(4)*xp(4) -y(4)*xp(4) -xp(4);
%             0 0 0 x(4) y(4) 1 -x(4)*yp(4) -y(4)*yp(4) -yp(4);
%             ];

        [~,~,V] = svd(P,0);
        H = reshape(V(:, end), [3,3])';
        
        D2_new = H*D1;
        
        D2_new(1,:) = D2_new(1,:)./D2_new(3,:);
        D2_new(2,:) = D2_new(2,:)./D2_new(3,:);
        D2_new(3,:) = D2_new(3,:)./D2_new(3,:);
        SD = sum((D2_new - D2).^2);
        inliers = find(SD < threshold);
        num_inlier = length(inliers) ;
        if (num_inlier > best_n_inlier)
            best_H = H;
            best_n_inlier = num_inlier;
            point1 = D1(1:2,inliers);
            point2 = D2(1:2,inliers);
            in_res = SD(inliers);
        end
    end
    H = best_H;
    num_inlier = best_n_inlier;
    
end