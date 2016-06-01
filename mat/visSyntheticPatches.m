function visSyntheticPatches()

clc; clear; close all;



% simulation image
im_sim = imread('~/Dropbox/NIPS16/figures/simulations/real_people.jpg');


anchors = [];
anchors(1,:) = [167, 138];
anchors(2,:) = [260, 160];
anchors(3,:) = [570, 170];
anchors(4,:) = [544, 262];
anchors(5,:) = [460, 130];
anchors(6,:) = [325, 235];
anchors(7,:) = [92,  169];
anchors(8,:) = [354,  91];
anchors(9,:) = [438, 230];
anchors(10,:) = [245, 105];

step = 1;
pos2d = dlmread(sprintf('~/develop/towncenter/src/pos2d_step%d.txt', step));
boundingboxes = dlmread(sprintf('~/develop/towncenter/src/boundingboxes_step%d.txt', step));

% figure(1); imshow(im_sim); hold on;
for i=1:size(anchors,1)
    iloc = find(ismember(pos2d,[anchors(i,1) anchors(i,2)], 'rows'));
    
%     rectangle('Position', boundingboxes(iloc, :), 'EdgeColor','r','LineWidth',1);
%     plot(anchors(i,1), anchors(i,2), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
%     text(anchors(i,1), anchors(i,2), sprintf('[ %d- y:%d,x:%d]', i,anchors(i,2),anchors(i,1)));

    box = boundingboxes(iloc,:);
    box(3) = box(3) - 1;
    box(4) = box(4) - 1;
    
    im_crpd = imcrop(im_sim, box);
    figure(1); imshow(im_crpd);

    % save
    fname = sprintf('somepatches/%d', i);
    print(fname, '-depsc');
end

end


