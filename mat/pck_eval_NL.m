function pck = pck_eval_NL(pred, gt)
assert(numel(pred) == numel(gt));
nJoints = 14;

pck_cnt = 0;
num_occ = 0;
for i=1:numel(gt)
    hsize = sqrt(sum((gt(i).point(1,:) - gt(i).point(2,:)).^2));
    for j=1:nJoints
        if gt(i).occ(j) == 1
            num_occ = num_occ+1;
        else
            d = sqrt(sum((gt(i).point(j,:)-pred(i).point(j,:)).^2));
            if d <= hsize*0.5
                pck_cnt = pck_cnt + 1;
            end
        end
    end
end

pck = ( pck_cnt / (nJoints * numel(gt) - num_occ) ) * 100;

end