%- visualize_prediction.m

clc;clear;

%% load data
task = 'PR_multi';
if strcmp (task, 'PR_full')
    fpath = '../save/PR_full/option,nEpochs=10000,t=PR_full/t_WedFeb1723:26:032016';
    fname_data = 'dataset/testdata.mat';
    fname_pred = fullfile(fpath, 'pred_te_PR_full.mat');
            
elseif strcmp (task, 'PR_multi')
    datatype = 'r27';     % s2000, r27
    kind  = 'test';
    fpath = '../save/testdir';
%     fpath = '../save/PR_multi/option,nEpochs=2000,t=PR_multi/t_TueFeb1614:05:152016';
%     fpath = '../save/PR_multi/option,t=PR_multi/t_ThuFeb1113:46:352016';
%     fpath = '../save/PR_multi/tmp';
    fname_data = fullfile(fpath, sprintf('testdata_%s_%s.mat',task, datatype));
    fname_pred = fullfile(fpath, sprintf('pred_%s_%s_%s.mat',kind, task, datatype));
            
elseif strcmp (task, 'PR_fcn')
    datatype = 'r27';
    kind  = 'test';
    fpath = '../save/testdir';
    fname_data = fullfile(fpath, sprintf('testdata_%s_%s.mat',task,datatype));
    fname_pred = fullfile(fpath, sprintf('pred_%s_%s_%s.mat', kind, task, datatype));
%     fname_hmap = fullfile(fpath, sprintf('gt_%s_%s_heatmap.mat', kind, task));
% 
%     hmap = load(fname_hmap); hmap = hmap.x;
%     hmap = permute(hmap, [3 4 2 1]);    
end

load(fname_data);
pred = load(fname_pred); pred = pred.x;
data = permute(data,[3 4 2 1]);


% %% load read data for test
% for i = 1:27
%     data_real{i}.img = imread(sprintf('cropped/%d.png',i));
%     data_real{i}.joint = dlmread(sprintf('cropped/joints/joints2d_%dth.txt',i));
% end



%% settings
resz = 4;
nJoints = 14;

ca = [];
gt = [];


%% visualize
if 1
for iSmp = 1:size(data,4) 
    img = data(:,:,:,iSmp);
    img = imresize(img, resz);
    
    joints_gt = reshape(label(iSmp, :), [2,nJoints]);
    joints_gt = joints_gt'.*repmat(resz*[64,128],[nJoints,1]);
    joints_pred = reshape(pred(iSmp, :), [2,nJoints]);
    joints_pred = joints_pred'.*repmat(resz*[64,128],[nJoints,1]);
    disp([joints_gt, joints_pred]);
    
    %- save to ca and gt for PCKh evaluation
    ca(iSmp).point = joints_pred;
    gt(iSmp).point = joints_gt;
    
    clf;figure(1); set(gcf, 'Position', [600,800,600,1200]);
    imshow(img); hold on;
    plot(joints_gt(:,1), joints_gt(:,2), 'b*', 'MarkerSize', 5);
    plot(joints_pred(:,1), joints_pred(:,2), 'r*', 'MarkerSize', 5);
    draw_sticks_joints(joints_gt, nJoints, 'b');
    draw_sticks_joints(joints_pred, nJoints, 'r');
    drawnow;
        
%     if strcmp(task, 'PR_fcn')
%         figure(2); set(gcf, 'Position', [600,800,600,1200]);
%         subplot(4,4,1); imshow(imresize(img,0.25));
%         for i = 1:14
%             hm = hmap(:,:,i,iSmp);
%             subplot(4,4,i+1); imagesc(hm); axis image; colormap gray;
%         end 
%     end

end
end


%% EVALUATE (PCKh, PCP)
ca = [];
gt = [];
for iSmp = 1:size(data,4)

    joints_gt = reshape(label(iSmp, :), [2,nJoints]);
    joints_gt = joints_gt'.*repmat(resz*[64,128],[nJoints,1]);
    joints_pred = reshape(pred(iSmp, :), [2,nJoints]);
    joints_pred = joints_pred'.*repmat(resz*[64,128],[nJoints,1]);
    
    gt(iSmp).quality = 1;
    gt(iSmp).state = ones(14,1);
    gt(iSmp).point = joints_gt([1 2 3 9 4 10 5 11 6 12 7 13 8 14],:);
    ca(iSmp).point = joints_pred([1 2 3 9 4 10 5 11 6 12 7 13 8 14],:);

end
[pck1,pck2,pck3] = pck_eval(ca,gt,0.5,'a','h');
[pcp1,pcp2,pcp3] = pcp_eval(ca,gt,0.5,'a');
fprintf('PCKh: %.2f \n', pck1);
fprintf('PCP: %.2f \n', pcp1);
