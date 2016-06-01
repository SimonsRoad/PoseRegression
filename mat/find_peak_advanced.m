function [joint,occ] = find_peak_advanced(hmap, nJoints, headsz)

% assert(size(hmap,3) == nJoints);
joint = [];
occ = [];
maxvals = [];
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
    maxvals(i) = max_val;
end



%% post processing
% for ankle, we perform 
% 0) check if left and right are pointing to the same location
% 1) compare left and right
% 2) choose the bigger one as the right pick
% 3) for the other one, perform non-maximum suppression based on the head
% size, and then choose the next max

% Joint Numbers: 
% lank: 8, rank: 14
% lkne: 7, rkne: 13 
% lkne_lank: 21, rkne_rank: 27 

% ankle
[joint, maxvals] = postprocess_differentiate_ankle(joint, hmap, maxvals, 8, 14, headsz);

% knee
[joint, maxvals] = postprocess_differentiate_knee(joint, hmap, maxvals, 7, 13, headsz);

% midpoints: kne_ank
[joint, maxvals] = postprocess_differentiate_ankle(joint, hmap, maxvals, 21, 27, headsz);

joint = postprocess_straightlegs(joint, hmap, maxvals, headsz);

end

function [loc,maxval] = findpeak_sub(hmap)
[maxval,idx] = max(hmap(:));
[y,x] = ind2sub(size(hmap), idx);
loc = [x,y];
end

function hmap = nonmaxsup(hmap, loc, headsz, beta)
% create circular mask
cx = loc(1); 
cy = loc(2);
ix = size(hmap,2);
iy = size(hmap,1);
r = headsz*beta;
[x,y]=meshgrid(-(cx-1):(ix-cx),-(cy-1):(iy-cy));
c_mask=((x.^2+y.^2)<=r^2);
% figure;imshow(c_mask); 

% suppression
hmap(c_mask) = 0;
end

function [joint, maxvals] = postprocess_differentiate_knee(joint, hmap, maxvals, nl, nr, headsz)
% head size parameters
alpha = 0.2;
beta  = 0.5;
gamma = 0.25;    % threshold for the minimum confidence value

if (abs(joint(nl,1)-joint(nr,1)))<headsz*alpha && (abs(joint(nl,2)-joint(nr,2)))<headsz*alpha   % points to the same location
    if maxvals(nl) > maxvals(nr)
        hmap_sup = nonmaxsup(hmap(:,:,nr), joint(nr,:), headsz, beta);
        [peakloc, peakval] = findpeak_sub(hmap_sup);
        s_rel = abs(maxvals(nr)-peakval)/abs(maxvals(nr));
        if s_rel < gamma
            joint(nr,:) = peakloc;
            maxvals(nr) = peakval;
        end
    elseif maxvals(nl) < maxvals(nr)
        hmap_sup = nonmaxsup(hmap(:,:,nl), joint(nl,:), headsz, beta);
        [peakloc, peakval] = findpeak_sub(hmap_sup);
        s_rel = abs(maxvals(nl)-peakval)/abs(maxvals(nl));
        if s_rel < gamma
            joint(nl,:) = peakloc;
            maxvals(nl) = peakval;
        end
    elseif maxvals(nl) == maxvals(nr)
        hmap_sup1 = nonmaxsup(hmap(:,:,nl), joint(nl,:), headsz, beta);
        hmap_sup2 = nonmaxsup(hmap(:,:,nr), joint(nr,:), headsz, beta);
        [max1,~] = max(hmap_sup1);
        [max2,~] = max(hmap_sup2);
        if max1 > max2
            [peakloc, peakval] = findpeak_sub(hmap_sup1);
            s_rel = abs(maxvals(nl)-peakval)/abs(maxvals(nl));
            if s_rel < gamma
                joint(nl,:) = peakloc;
                maxvals(nl) = peakval;
            end
        elseif max1 > max2
            [peakloc, peakval] = findpeak_sub(hmap_sup2);
            s_rel = abs(maxvals(nr)-peakval)/abs(maxvals(nr));
            if s_rel < gamma
                joint(nr,:) = peakloc;
                maxvals(nr) = peakval;
            end
        else
            error('currently not implemented yet..!');
        end
    else
        error('This cannot happen!');
    end
end
end

function [joint, maxvals] = postprocess_differentiate_ankle(joint, hmap, maxvals, nl, nr, headsz)
% head size parameters
alpha = 0.2;
beta  = 0.5;
gamma = 0.1;    % threshold for the minimum confidence value

if (abs(joint(nl,1)-joint(nr,1)))<headsz*alpha && (abs(joint(nl,2)-joint(nr,2)))<headsz*alpha   % points to the same location
    if maxvals(nl) > maxvals(nr)
        hmap_sup = nonmaxsup(hmap(:,:,nr), joint(nr,:), headsz, beta);
        [peakloc, peakval] = findpeak_sub(hmap_sup);
        if peakval > gamma
            joint(nr,:) = peakloc;
            maxvals(nr) = peakval;
        end
    elseif maxvals(nl) < maxvals(nr)
        hmap_sup = nonmaxsup(hmap(:,:,nl), joint(nl,:), headsz, beta);
        [peakloc, peakval] = findpeak_sub(hmap_sup);
        if peakval > gamma
            joint(nl,:) = peakloc;
            maxvals(nl) = peakval;
        end
    elseif maxvals(nl) == maxvals(nr)
        hmap_sup1 = nonmaxsup(hmap(:,:,nl), joint(nl,:), headsz, beta);
        hmap_sup2 = nonmaxsup(hmap(:,:,nr), joint(nr,:), headsz, beta);
        [max1,~] = max(hmap_sup1);
        [max2,~] = max(hmap_sup2);
        if max1 > max2
            [peakloc, peakval] = findpeak_sub(hmap_sup1);
            if peakval > gamma
                joint(nl,:) = peakloc;
                maxvals(nl) = peakval;
            end
        elseif max1 > max2
            [peakloc, peakval] = findpeak_sub(hmap_sup2);
            if peakval > gamma
                joint(nr,:) = peakloc;
                maxvals(nr) = peakval;
            end
        else
            error('currently not implemented yet..!');
        end
    else
        error('This cannot happen!');
    end
end
end

function joint = postprocess_straightlegs(joint, hmap, maxvals, headsz)
% lank: 8, rank: 14
% lkne: 7, rkne: 13 
% lkne_lank: 21, rkne_rank: 27 

lkne = joint(7,:);
rkne = joint(13,:);
lmid = joint(21,:);
rmid = joint(27,:);
lank = joint(8,:);
rank = joint(14,:);


% angles for [ lll, llr, lrl, lrr, rll, rlr, rrl, rrr ]
a_lll = angleBetweenVectors(lkne-lmid,lank-lmid);
a_llr = angleBetweenVectors(lkne-lmid,rank-lmid);
a_lrl = angleBetweenVectors(lkne-rmid,lank-rmid);
a_lrr = angleBetweenVectors(lkne-rmid,rank-rmid);
a_rll = angleBetweenVectors(rkne-lmid,lank-lmid);
a_rlr = angleBetweenVectors(rkne-lmid,rank-lmid);
a_rrl = angleBetweenVectors(rkne-rmid,lank-rmid);
a_rrr = angleBetweenVectors(rkne-rmid,rank-rmid);
angs = [a_lll a_llr a_lrl a_lrr a_rll a_rlr a_rrl a_rrr];

if norm(lkne-rkne) <= 1e-2
    angs = [a_lll a_llr a_lrl a_lrr];
%     fprintf('a_lll: %.1f \n', a_lll);
%     fprintf('a_llr: %.1f \n', a_llr);
%     fprintf('a_lrl: %.1f \n', a_lrl);
%     fprintf('a_lrr: %.1f \n', a_lrr);

    all_joint{1} = [lkne;lmid;lank];
    all_joint{2} = [lkne;lmid;rank];
    all_joint{3} = [lkne;rmid;lank];
    all_joint{4} = [lkne;rmid;rank];
    
    [angs_srt, indices] = sort(angs, 'descend');
    cand = all_joint(indices(1:2));
    
    for i = 1:2
        if indices(i) == 1
            joint([7 21 8],:) = all_joint{1};
        end
        if indices(i) == 2
            joint([7 21 14],:) = all_joint{4};
        end
        if indices(i) == 3
            joint([7 27 8],:) = all_joint{1};
        end
        if indices(i) == 4
            joint([7 27 14],:) = all_joint{4};
        end
    end
    
else
    angs = [a_lll a_llr a_lrl a_lrr a_rll a_rlr a_rrl a_rrr];
%     fprintf('a_lll: %.1f \n', a_lll);
%     fprintf('a_llr: %.1f \n', a_llr);
%     fprintf('a_lrl: %.1f \n', a_lrl);
%     fprintf('a_lrr: %.1f \n', a_lrr);
%     fprintf('a_rll: %.1f \n', a_rll);
%     fprintf('a_rlr: %.1f \n', a_rlr);
%     fprintf('a_rrl: %.1f \n', a_rrl);
%     fprintf('a_rrr: %.1f \n', a_rrr);

    all_joint{1} = [lkne;lmid;lank];
    all_joint{2} = [lkne;lmid;rank];
    all_joint{3} = [lkne;rmid;lank];
    all_joint{4} = [lkne;rmid;rank];
    all_joint{5} = [rkne;lmid;lank];
    all_joint{6} = [rkne;lmid;rank];
    all_joint{7} = [rkne;rmid;lank];
    all_joint{8} = [rkne;rmid;rank];
    
    [angs_srt, indices] = sort(angs, 'descend');
    cand = all_joint(indices(1:2));
    
    for i = 1:2
        if indices(i) == 1
            joint([7 21 8],:) = all_joint{1};
        end
        if indices(i) == 2
            joint([7 21 14],:) = all_joint{1};
        end
        if indices(i) == 3
            joint([7 27 8],:) = all_joint{1};
        end
        if indices(i) == 4
            joint([7 27 14],:) = all_joint{8};
        end
        if indices(i) == 5
            joint([13 21 8],:) = all_joint{1};
        end
        if indices(i) == 6
            joint([13 21 14],:) = all_joint{8};
        end
        if indices(i) == 7
            joint([13 27 8],:) = all_joint{8};
        end
        if indices(i) == 8
            joint([13 27 14],:) = all_joint{8};
        end
    end    
end


  

end

function angle = angleBetweenVectors(v1, v2)
costheta = dot(v1,v2)/(norm(v1)*norm(v2));
theta = acos(costheta);
angle = radtodeg(theta);
end
