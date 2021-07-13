function mse=fH_MSE(H,points1,points2)
   cor_points2=H*points1;
   for i = 1 : 3
            cor_points2(i,:) = cor_points2(i,:)./cor_points2(3,:);
   end
   
   [m,n]=size(points2);
   mse = (norm(cor_points2 - points2).^2)/(n);
end