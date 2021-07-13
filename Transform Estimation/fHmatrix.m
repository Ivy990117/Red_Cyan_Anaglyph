function[H, num_inlier, point1, point2, in_res,MSE] = fHmatrix(D1, D2, threshold, num, iternations)
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
        A=zeros(2*num,9);
        for irow=1:num
            %odd rows
            A(2*irow-1,:)=[x(irow), y(irow) 1 0 0 0 -x(irow)*xp(irow) -y(irow)*xp(irow) -xp(irow)];
            %even rows
            A(2*irow,:)=[0 0 0 x(irow) y(irow) 1 -x(irow)*yp(irow) -y(irow)*yp(irow) -yp(irow)];
        end
          
        %compute Homography matrix H (from D2 to D1)
        
%         [~,~,V] = svd(A,0);
%         H = reshape(V(:, end), [3,3])';
        
        [~,~,V] = svd(A'*A,0);
        H = reshape(V(:, end)/V(9,9), [3,3])';
        %homogeneous coordinates of reconstructed image 2 from image 1
        D2_new = H*D1;
        
        %image coordinates of reconstructed image 2 from image 1
        for i = 1 : 3
            D2_new(i,:) = D2_new(i,:)./D2_new(3,:);
        end
   
        %compute the differences between D2 and D2_new
        SD = sum((D2_new - D2).^2);
        inliers = find(SD < threshold);
        num_inlier = length(inliers) ;
        
        %evaluate the effectiveness of H and find the best one
        if (num_inlier > best_n_inlier)
            best_H = H;
            best_n_inlier = num_inlier;
            point1 = D1(1:2,inliers);
            point2 = D2(1:2,inliers);
            in_res = SD(inliers);
        end
    end
    
    H = best_H;
    MSE=fH_MSE(H,D1,D2);
    num_inlier = best_n_inlier;    
end