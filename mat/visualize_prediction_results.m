% visualize_prediction_results.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
%
clear; close all;


%% settings
addpath('~/Downloads/convolutional-pose-machines-release/testing/');

% select location
pck_mean_all = [];
numimagesall = 0;
for selLoc = [1]
    % anchor locations for [towncenter dataset]
%     YX = [...
%         138 167;
%         160 260;
%         170 570;
%         262 544;
%         130 460;
%         235 325;
%         169 92;
%         91 354;
%         230 438;
%         105 245;
%         999 999;
%         138 167; 
%         138 167;
%         138 167;
%         ];
%     if selLoc < 11 
%         ablative = 'no';
%     elseif selLoc == 11
%         ablative = 'generic';
%     elseif selLoc == 12 
%         ablative = 'noprior';
%     elseif selLoc == 13
%         ablative = 'more';
%     elseif selLoc == 14
%         ablative = 'multiloss';
%     else
%         error('invalid location');
%     end
%     datasetname = 'towncenter';
%     datasettype = 'rTest';
    
    % anchor locations for [pet2006 dataset]
    YX = [240 150;
        270 550;
        250 340;
        420 130;
        240 150;
        240 150;
        240 150];
    if selLoc < 5
        ablative = 'no';
    elseif selLoc == 5
        ablative = 'CF1';
    elseif selLoc == 6
        ablative = 'CF2';
    elseif selLoc == 7
        ablative = 'nosegcen';
    else
        error('invalid location');
    end
    datasetname = 'pet2006';
    datasettype = 'rTest';
    
    y = YX(selLoc,1);
    x = YX(selLoc,2);
    
    
    %% load data
    dataset = load_dataset(x,y, datasetname, datasettype);
    dataset = load_predictions(dataset, datasettype, ablative);
    
    
    %% compute PCK
%     nJoints = 14;
%     pred = []; 
%     gt = [];
%     for i=1:numel(dataset)
%         pred(i).point = dataset(i).point_pred(1:nJoints,:);
%         gt(i).point = dataset(i).point(1:nJoints,:);
%         gt(i).occ = dataset(i).occ(1:nJoints);
%     end
%     
%     pck_all = [];
%     normscalor = 0:0.05:0.5;
%     for i=1:numel(dataset)
%         pck_norms = [];
%         for j=normscalor
%             pck = pck_eval_NL(pred(i), gt(i), j);
%             pck_norms = [pck_norms pck];
%         end
%         pck_all = [pck_all; pck_norms];
%     end
%     pck_mean = mean(pck_all);
%     assert(numel(pck_mean)==numel(normscalor));
% %     fprintf('PCK (all images): \n');
%     for i=1:numel(normscalor)
%         fprintf('%.2f\t', pck_mean(i));
%     end; fprintf('\n');
%     
%     pck_mean_all = [pck_mean_all; pck_mean*numel(dataset)];
%     numimagesall = numimagesall + numel(dataset);
    
    %% visualize
    parts.name = {'head_neck', ...
        'neck_lsho', 'lsho_lelb', 'lelb_lwr', 'lhip_lkne', 'lkne_lank', ...
        'neck_rsho', 'rsho_relb', 'relb_rwr', 'rhip_rkne', 'rkne_rank'};
    parts.seq = {[1 2], ...
        [2 3], [3 4], [4 5], [6 7], [7 8], ...
        [2 9], [9 10], [10 11], [12 13], [13 14]};
%     CM = jet(numel(parts.seq));
    CM(1,:)  = [0 0 0.6667];
    CM(2,:)  = [1 1 0];
    CM(3,:)  = [1 1 0];
    CM(4,:)  = [1 1 0];
    CM(5,:)  = [0 0.3333 1];
    CM(6,:)  = [0 0.3333 1];
    CM(7,:)  = [0 0.6667 1];
    CM(8,:)  = [0 0.6667 1];
    CM(9,:)  = [0 0.6667 1];
    CM(10,:) = [1 0.3333 0];
    CM(11,:) = [1 0.3333 0];
    for i=1:numel(dataset)
        
        % high quality image
        img = imread(strrep(dataset(i).im, 'LQ/pos', 'HQ/allpos'));
        
        %     seg = dataset(i).jsc(:,:,28);
        %     seg(seg<0) = 0;
        %     seg = seg / max(seg(:));
        %     mapIm = mat2im(seg, jet(100), [0 1]);
        %     img_aug = mapIm*0.5 + (im2single(img))*0.5;
        seg = dataset(i).jsc(:,:,28);
        mapIm(:,:,2) = zeros(size(seg));
        mapIm(:,:,1) = seg;
        mapIm(:,:,3) = zeros(size(seg));
        img_aug = mapIm*0.5 + (im2single(img))*1.0;
        
        figure(1); imshow(img_aug); hold on;
        set(gcf,'InvertHardCopy','off');
        
        % predicted joints
        joints = dataset(i).point_pred;
        nJoints = 14;
        for j=1:nJoints
            plot(joints(j,1),joints(j,2),'go','MarkerSize', 3, 'LineWidth',1.5);
        end
        % For joints:   saved MarkerSize:7 LineWidth: 1.5 
        % For Skeleton: saved LineWidth: 4
        
        % skeleton
        for j=1:numel(parts.seq)
            x1 = joints(parts.seq{j}(1),1);
            x2 = joints(parts.seq{j}(2),1);
            y1 = joints(parts.seq{j}(1),2);
            y2 = joints(parts.seq{j}(2),2);
            plot([x1,x2], [y1,y2], 'color', CM(j,:), 'LineWidth', 2);
        end
        fname = sprintf('qualitative/%s/%s/y%03d_x%d/frm%04d_seg_j_sk', datasetname,datasettype,dataset(1).y,dataset(1).x, dataset(i).frm);
        print(fname, '-depsc');
        
        hold off;
    end
    
%     clear;
end

fprintf('\n');
pck_final = sum(pck_mean_all,1) / numimagesall;
for i=1:numel(pck_final)
    fprintf('%.2f\t', pck_final(i));
end; fprintf('\n');
    










