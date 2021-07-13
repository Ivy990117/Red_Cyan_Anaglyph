%formulate windows w.r.t. each keypoints
function [neighbourhood1,neighbourhood2] = WindowGeneration(points_left, points_right,l,image1,image2)
    neighbourhood1 = cell(length(points_left),1);
    neighbourhood2 = cell(length(points_right),1);
    for i = 1:size(neighbourhood1)
       neighbourhood1{i} = reshape(image1(points_left(i,1)-l:points_left(i,1)+l,points_left(i,2)-l:points_left(i,2)+l),[(l*2+1)^2 1]);
    end
    for i = 1:size(neighbourhood2)
        neighbourhood2{i} = reshape(image2(points_right(i,1)-l:points_right(i,1)+l,points_right(i,2)-l:points_right(i,2)+l),[(l*2+1)^2 1]);
    end
end
