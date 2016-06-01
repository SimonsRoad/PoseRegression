function [joint,occ] = find_peak(hmap, nJoints)

% assert(size(hmap,3) == nJoints);
joint = [];
occ = [];
for i = 1:nJoints
    tmp = hmap(:,:,i);
    [max_val,idx] = max(tmp(:));
    [y,x] = ind2sub(size(tmp), idx);
    joint(i,:) = [x,y];
    if max_val < 0.9 
        occ(i) = 1;
    else
        occ(i) = 0;
    end
    if i==8
end

end