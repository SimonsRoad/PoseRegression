function run_poseestimation(testdata, posemethod)

switch(posemethod)
    case 'CPM'
        run_CPM(testdata);
    case 'IEF'
        run_IEF(testdata);
    otherwise
        error('Check available methods for pose estimation!');
end

end


%% CPM
function run_CPM(testdata)

addpath('~/Downloads/convolutional-pose-machines-release/testing/src');
addpath('~/Downloads/convolutional-pose-machines-release/testing/util');
addpath('~/Downloads/convolutional-pose-machines-release/testing/util/ojwoodford-export_fig-5735e6d/');
param = config();
interestPart = 'Lwri'; % to look across stages. check available names in config.m
fprintf('Description of selected model: %s \n', param.model(param.modelID).description);


%% model loading
model = param.model(param.modelID);
boxsize = model.boxsize;
np = model.np;
nstage = model.stage;
net = caffe.Net(model.deployFile, model.caffemodel, 'test');


%% NL
nImages = numel(testdata);
% nImages = 2;
prediction_all = [];
time_total = 0;
for i = 1:nImages
    % select image
%     fprintf('CPM: processing image#: %d \n',i);
    
    
    %% core: apply model on the image, to get heat maps and prediction coordinates
    tic;
    [heatMaps, prediction] = applyModel2(testdata(i).im, param, testdata(i).box, net);
    time = toc;
    prediction_all(i).point = prediction;
    time_total = time_total + time;
    
    %% visualize, or extract variable heatMaps & prediction for your use
    %     visualize('tmp.jpg', heatMaps, prediction, param, rectangle, interestPart);
    
end
fprintf('total processing time (CPM): %.4f \n', time_total);
fprintf('  avg processing time (CPM): %.4f \n', time_total/nImages);

%% PCK
% dataset_pred.point = prediction([1,2,6,7,8,12,13,14,3,4,5,9,10,11],:);
nJoints = 14;
pck_all = 0;
for i=1:nImages
    tmp = prediction_all(i).point;
    prediction_all(i).point = tmp([1,2,6,7,8,12,13,14,3,4,5,9,10,11],:);
    testdata(i).point = testdata(i).point(1:nJoints,:);
    testdata(i).occ = testdata(i).occ(1:nJoints);
    
    % compute PCK
%     pck = pck_eval(prediction_all(i), testdata(i), 0.5, 'a', 'h');
    pck = pck_eval_NL(prediction_all(i), testdata(i));
    pck_all = pck_all + pck;
%     fprintf('PCK (%dth image): %.2f \n', i, pck);
end
% fprintf('PCK (all images): %.2f \n', pck_all/nImages);

end

%% IEF
function run_IEF(testdata)
addpath('~/Downloads/IEF/IEF-release-v1');
run('~/Downloads/IEF/IEF-release-v1/matconvnet-1.0-beta14/matlab/vl_setupnn.m');

% load ConvNets
load('~/Downloads/IEF/IEF-release-v1/models/scalesel-vggs-epoch-14.mat', 'net');
net_scale = net;
clear net;
net_scale.layers(end) = []; % remove loss layer
net_scale = vl_simplenn_move(net_scale,'gpu');

% trained for 12 iterations in each stage, 13 in the last one
% (twice as many as in the November 2015 arxiv)
% gets 81.8 on MPII validation without ground truth scale (up from 81.0 in paper).

load('~/Downloads/IEF/IEF-release-v1/models/IEF-googlenet-epoch-49.mat', 'net', 'params');
net = dagnn.DagNN.loadobj(net);
net.move('gpu') ;

nImages = numel(testdata);
prediction_all = [];
time_total = 0;
for i=1:nImages
    tic;
%     fprintf('IEF: processing image %d.. \n', i);
    I = imread(testdata(i).im);
    
    %% NL
    pt_in_torso = [testdata(i).box(1)+(testdata(i).box(3)/2) testdata(i).box(2)+(testdata(i).box(4)/2)];
    
    img_side = 256;
    crop_side = 224;
    lambda = [1.4142 1.1892 1 0.8409 0.7071 0.5946 0.5 0.4204 0.3536  0.2973];
    [all_I, cropinfo] = crop_img(I, pt_in_torso, img_side, lambda);
    
    % select scale automatically
    all_I_stack = zeros(img_side,img_side,3,numel(lambda), 'single');
    for j=1:numel(lambda)
        all_I_stack(:,:,:,j) = bsxfun(@minus, single(all_I{j}), net_scale.normalization.averageImage(1,1,:));
    end
    
    res_scale = vl_simplenn(net_scale, gpuArray(all_I_stack), [], [], ...
        'disableDropout', true, ...
        'conserveMemory', true);
    
    [~,best_scale] = max(squeeze(res_scale(end).x));
    
    selI = all_I{best_scale};
    
    % get center patch
    dif = (img_side-crop_side)/2;
    range = dif+1:img_side-dif;
    selI = selI(range, range, :);
    
    % predict pose
     
    selI = gpuArray(single(selI));
    selI = bsxfun(@minus, selI, net.meta.normalization.averageImage(1,1,:));

    pose = params.seed_pose - dif;
    pose(17,:) = ([crop_side crop_side]./2); % marking point is in center of cropped image
    all_poses = pose;
    
    N_STEPS = 5;
    
    target_dim = 141;
    
    for j=1:N_STEPS
        pose = all_poses(:,:,end);
        
        inputs = {'image', selI, 'kp_pos', pose, 'label', zeros(32,1)};
        net.vars(target_dim).precious = 1;
        net.eval(inputs) ;
        corrections = gather(squeeze(net.vars(target_dim).value));
        corrections = reshape(corrections, [16 2]);
        
        pose(1:16,:) = pose(1:16,:) + params.MAX_STEP_NORM*corrections;
        all_poses = cat(3,all_poses, pose);
    end
    time = toc;
    time_total = time_total + time;
    
    if 0
        clf; close all;
        sc(selI); hold on;
        plot_pose_stickmodel(pose);
    end
    
    %% restore joints from predicted pose
    % 1. firstly convert back from crop_side to img_side
    pose = pose + dif;
    % 2. secontly convert back using how it's cropped to crop_side in the
    % first place
    prediction_all(i).point = convert_pose_original(pose, cropinfo(best_scale));
    % sanity check
    if 0
        figure(2); imshow(I); hold on;
        scatter(prediction_all(i).point(:,1), prediction_all(i).point(:,2));
        hold off;
    end
end
fprintf('total processing time (IEF): %.4f \n', time_total);
fprintf('  avg processing time (IEF): %.4f \n', time_total/nImages);

%% PCK
nJoints = 14;
pck_all = 0;
for i=1:nImages
    tmp = prediction_all(i).point;
    prediction_all(i).point = tmp([10 9 14 15 16 4 5 6 13 12 11 3 2 1],:);    % IEF definition to mine
    testdata(i).point = testdata(i).point(1:nJoints,:);
    
    % compute PCK
%     pck = pck_eval(prediction_all(i), testdata(i), 0.5, 'a', 'h');
    pck = pck_eval_NL(prediction_all(i), testdata(i));
    pck_all = pck_all + pck;
%     fprintf('PCK (%dth image): %.2f \n', i, pck);
end
% fprintf('PCK (all images): %.2f \n', pck_all/nImages);



end

