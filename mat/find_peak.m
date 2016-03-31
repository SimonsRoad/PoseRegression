function joint = find_peak(hmap, nJoints)
assert(size(hmap,3) == nJoints);
joint = [];
for i = 1:nJoints
    tmp = hmap(:,:,i);
    [max_val,idx] = max(tmp(:));
    [y,x] = ind2sub(size(tmp), idx);
    joint(i,:) = [x,y];
end

end