function visDetectionPoseEstionSegmentation()
clear; close all; clc;
addpath('~/Downloads/convolutional-pose-machines-release/testing/');


% selLoc = 1;

for selLoc = 1:4
    
    [y,x] = chooseAnchorLocation(selLoc);
    datasetname = 'pet2006';
    datasettype = 'rTest';
    
    % load dataset
    dataset = load_dataset(x,y, datasetname, datasettype);
    
    % load prediction
    dataset = load_predictions(dataset, datasettype, 'no');
    
    % get bounding box (OURS)
    dataset = run_detection(dataset, 'OURS', datasetname, datasettype);
    
    % just for information
    d = load_detector_info(dataset(1).y,dataset(1).x, 'no');

    
    %% VISUALIZE [1. joints, 2. skeletons, 3. segmentation, 4. bbox]
    % draw onto the original resolution image
    % resize everything to fit the original resolution (x3 larger)
    % use original resolution image
    
    parts.name = {'head_neck', ...
        'neck_lsho', 'lsho_lelb', 'lelb_lwr', 'lhip_lkne', 'lkne_lank', ...
        'neck_rsho', 'rsho_relb', 'relb_rwr', 'rhip_rkne', 'rkne_rank', ...
        'lhip_rhip'};
    parts.seq = {[1 2], ...
        [2 3], [3 4], [4 5], [6 7], [7 8], ...
        [2 9], [9 10], [10 11], [12 13], [13 14], ...
        [6 12]};
    % CM(1,:)  = [0 0 0.6667];
    % CM(2,:)  = [1 1 0];
    % CM(3,:)  = [1 1 0];
    % CM(4,:)  = [1 1 0];
    % CM(5,:)  = [0 0.3333 1];
    % CM(6,:)  = [0 0.3333 1];
    % CM(7,:)  = [0 0.6667 1];
    % CM(8,:)  = [0 0.6667 1];
    % CM(9,:)  = [0 0.6667 1];
    % CM(10,:) = [1 0.3333 0];
    % CM(11,:) = [1 0.3333 0];
    % CM(12,:) = [0 1 0];
    % CM(13,:) = [0 1 0];
    CM = [0 0 1];
    for i=1:numel(dataset)
        
        if 1
        % high quality image
        img = imread(strrep(dataset(i).im, 'LQ/pos', 'HQ/allpos'));
        seg = dataset(i).jsc(:,:,28);
        mapIm = [];
        mapIm(:,:,2) = zeros(size(seg));
        mapIm(:,:,1) = seg;
        mapIm(:,:,3) = zeros(size(seg));
        img_aug = mapIm*1.0 + (im2single(img))*1.0;
        
        figure(1); imshow(img_aug); hold on;
        set(gcf,'InvertHardCopy','off');

        % predicted joints
        joints = dataset(i).point_pred;
        nJoints = 14;
        
        % draw skeleton
        for j=1:numel(parts.seq)
            x1 = joints(parts.seq{j}(1),1);
            x2 = joints(parts.seq{j}(2),1);
            y1 = joints(parts.seq{j}(1),2);
            y2 = joints(parts.seq{j}(2),2);
            %         plot([x1,x2], [y1,y2], 'color', CM(j,:), 'LineWidth', 2);
            plot([x1,x2], [y1,y2], 'color', CM, 'LineWidth', 2);
        end
        mhip = mean(joints([6,12],:));
        neck = joints(2,:);
        plot([mhip(1) neck(1)], [mhip(2) neck(2)], 'color', CM, 'LineWidth',2);
        
        % draw joints
        for j=1:nJoints
            plot(joints(j,1),joints(j,2),'go','MarkerSize', 3, 'LineWidth',1.5);
        end
        % For joints:   saved MarkerSize:7 LineWidth: 1.5
        % For Skeleton: saved LineWidth: 4
        % For bbox:     saved LineWidth: 3
        
        % bbox
        rectangle('Position', dataset(i).box, 'EdgeColor', [0.333 1 0.333], 'LineWidth', 6);
        
        
        % save
        fname = sprintf('qualitative/%s/%s/y%03d_x%d/frm%04d_seg_j_sk', datasetname,datasettype,dataset(1).y,dataset(1).x, dataset(i).frm);
        print(fname, '-dpng');

        
        hold off;
        end
        
        
        %% Load saved prediction results, resize and then re-save!!
        if 0
            fname = sprintf('qualitative/%s/%s/y%03d_x%d/frm%04d_seg_j_sk.png', datasetname,datasettype,dataset(1).y,dataset(1).x, dataset(i).frm);
            img = imread(fname);
            figure(1); imshow(img);
            
            % crop and resize
%             box = [427 117 767-427 606-117];
%             box = [434 126 760-434 589-126];
            box = [411 105 783-411 636-105];
            img_crop = imcrop(img, box);
            img_res = imresize(img_crop, [d.h d.w]);
%             figure(2); imshow(img_res);
            fname_new = sprintf('qualitative_new/%s/%s/y%03d_x%d/frm%04d_seg_j_sk.png', datasetname,datasettype,dataset(1).y,dataset(1).x, dataset(i).frm);
            imwrite(img_res, fname_new);
        end
        
    end

end

end


function [y,x] = chooseAnchorLocation(selLoc)
% YX = [...
%     138 167;
%     160 260;
%     170 570;
%     262 544;
%     130 460;
%     235 325;
%     169 92;
%     91 354;
%     230 438;
%     105 245;
%     999 999;
%     138 167;
%     138 167;
%     138 167;
%     ];
% y = YX(selLoc,1);
% x = YX(selLoc,2);

YX = [240 150;
    270 550;
    250 340;
    420 130;
    240 150;
    240 150;
    240 150];
y = YX(selLoc,1);
x = YX(selLoc,2);


end
