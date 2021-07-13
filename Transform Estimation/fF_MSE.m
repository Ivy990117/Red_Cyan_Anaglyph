function mse=fF_MSE(F,points1,points2)
   [~,n]=size(points2);
   points1=[points1' ones(n,1)];
   points2=[points2; ones(1,n)];
   temp=points1*F*points2;
   [~,m]=size(temp);   
   mse = (norm(temp).^2)/(m*m);
end