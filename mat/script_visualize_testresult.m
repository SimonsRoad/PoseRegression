% script_visualize_testresult.m 

clc; clear; close all;

testtype   = 'rTest';
% pathtodata = ['../save/PR_fcn/option/t_SunMar2721:48:402016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_FriApr809:10:502016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_ThuMar3112:33:322016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option/t_WedApr1308:12:392016/results/', testtype];
% pathtodata = ['../save/PR_fcn/option,LR=0.01/t_WedApr1320:28:332016/results/', testtype];
pathtodata = ['../save/PR_fcn/option/t_SunApr1709:33:532016/results/', testtype];


mNum    = 2;
nData   = 25;
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
fname_jsc_gt = sprintf('jsc_gt_model%d.mat',mNum);
load(fullfile(pathtodata,fname_jsc_gt));
x = permute(x, [3,4,2,1]);
assert(size(x,4) == nData);
for i = 1:nData
    dataset_gt(i).jsc   = x(:,:,:,i);
    dataset_gt(i).point = find_peak(dataset_gt(i).jsc(:,:,1:nJoints), nJoints);         % joint location
end

% sanity check
if 0
    randidx = randi(nData);
    visualize_jsc(dataset_gt(i).img, dataset_gt(i).jsc);
end

%% load pred label (jsc, .mat file)
fname_jsc_pred = sprintf('jsc_pred_model%d.mat',mNum);
load(fullfile(pathtodata,fname_jsc_pred));
x = permute(x, [3,4,2,1]);
assert(size(x,4) == nData);
for i = 1:nData
    dataset_pred(i).jsc   = x(:,:,:,i);
    dataset_pred(i).point = find_peak(dataset_pred(i).jsc(:,:,1:nJoints), nJoints);       % joint location
end


%% evaluation
pck = pck_eval(dataset_pred, dataset_gt, 0.5, 'a', 'h');
fprintf('PCK: %.2f \n', pck);


% visualize
for i = 1:nData
    visualize_jsc(dataset_gt(i).img, dataset_pred(i).jsc, dataset_gt(i).jsc, nJoints);
    title(pck_eval(dataset_pred(i), dataset_gt(i), 0.5, 'a', 'h'));
end






