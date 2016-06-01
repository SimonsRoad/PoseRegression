function visWholeScenePredictionResults()
clc; clear; close all;

% patches
patches = [];
for i = 1:10
    patches(i).im = sprintf('predPatches/%d.png',i);
end

% background image
background = imread('~/develop/towncenter/data/background.jpg');
background = background*0.8;


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

tightbox = [];  % x1 y1 x2 y2
tightbox(1,:) = [139 81 169 145];
tightbox(2,:) = [238 87 268 166];
tightbox(3,:) = [572 88 606 153];
tightbox(4,:) = [525 164 567 254];
tightbox(5,:) = [426 65 456 132];
tightbox(6,:) = [276 169 316 264];
tightbox(7,:) = [70 109 110 179];
tightbox(8,:) = [344 42 370 96];
tightbox(9,:) = [393 154 433 240];
tightbox(10,:) = [242 38 268 90];


step = 1;
pos2d = dlmread(sprintf('~/develop/towncenter/src/pos2d_step%d.txt', step));
boundingboxes = dlmread(sprintf('~/develop/towncenter/src/boundingboxes_step%d.txt', step));

for i=size(anchors,1):-1:1
    
    % area to be replaced
    iloc = find(ismember(pos2d,[anchors(i,1) anchors(i,2)], 'rows'));
    area = boundingboxes(iloc, :);
    
    % image to put
    img_put = imread(patches(i).im);
    
    % palette
    tmp = zeros(size(background));
    tmp(area(2):area(2)+area(4)-1, area(1):area(1)+area(3)-1, :) = img_put;
    
    % fine-level crop (only around bounding box)
    img_put_crop = tmp(tightbox(i,2):tightbox(i,4), tightbox(i,1):tightbox(i,3), :);
    
    
    % replace
    background(tightbox(i,2):tightbox(i,4), tightbox(i,1):tightbox(i,3), :) = img_put_crop;
    
    
    figure(1); imshow(background);
end

% figure(2); imshow(background); hold on;
% for i=1:size(anchors,1)
%     
%     % area to be replaced
%     iloc = find(ismember(pos2d,[anchors(i,1) anchors(i,2)], 'rows'));
%     area = boundingboxes(iloc, :);
%     
%     % draw a box to check..
%     rectangle('Position', area, 'EdgeColor','r','LineWidth',1);
%     plot(anchors(i,1), anchors(i,2), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
%     text(anchors(i,1), anchors(i,2), sprintf('[ %d- y:%d,x:%d]', i,anchors(i,2),anchors(i,1)));    
% end






% save 
fname = 'wholesceneprediction/result';
print(fname, '-depsc');

end