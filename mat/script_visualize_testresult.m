% script_visualize_testresult.m 

clc; clear; close all;

testtype   = 'rTest';
% pathtodata = ['../save/PR_fcn/option/t_SunMar2721:48:402016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_FriApr809:10:502016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_ThuMar3112:33:322016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_WedApr1308:12:392016/results/', testtype];
pathtodata = ['../save/PR_fcn/option,LR=0.01/t_WedApr1320:28:332016/results/', testtype];


mNum    = 2;
nData   = 22;
nJoints = 14;   % this can be 27 for synthetic data. For real it's 14.

dataset_gt   = [];
dataset_pred = [];
%% test image (.mat file)
fname_img = sprintf('img_model%d.mat',mNum);
load(fullfile(pathtodata,fname_img));
x = permute(x, [3,4,2,1]);
assert(size(x,4) == nData);
for i = 1:nData
    dataset_gt(i).img = x(:,:,:,i);
end


%% load gt label (j27 for synthetic or j14 for real)
fname_jsdc_gt = sprintf('jsdc_gt_model%d.mat',mNum);
load(fullfile(pathtodata,fname_jsdc_gt));
x = permute(x, [3,4,2,1]);
assert(size(x,4) == nData);
for i = 1:nData
    dataset_gt(i).jsdc  = x(:,:,:,i);
    dataset_gt(i).point = find_peak(dataset_gt(i).jsdc(:,:,1:nJoints), nJoints);         % joint location
end

% sanity check
if 0
    randidx = randi(nData);
    visualize_jsdc(dataset_gt(i).img, dataset_gt(i).jsdc);
end

%% load pred label (jsdc, .mat file)
fname_jsdc_pred = sprintf('jsdc_pred_model%d_1.mat',mNum);
load(fullfile(pathtodata,fname_jsdc_pred));
x = permute(x, [3,4,2,1]);
assert(size(x,4) == nData);
for i = 1:nData
    dataset_pred(i).jsdc  = x(:,:,:,i);
    dataset_pred(i).point = find_peak(dataset_pred(i).jsdc(:,:,1:nJoints), nJoints);       % joint location
end


%% evaluation
pck = pck_eval(dataset_pred, dataset_gt, 0.5, 'a', 'h');
fprintf('PCK: %.2f \n', pck);


% visualize
for i = 1:nData
    visualize_jsdc(dataset_gt(i).img, dataset_pred(i).jsdc, dataset_gt(i).jsdc, nJoints);
    title(pck_eval(dataset_pred(i), dataset_gt(i), 0.5, 'a', 'h'));
end






