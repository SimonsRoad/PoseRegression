function testdata = run_0detection(testdata, detectionmethod, datasetname, datasettype)
% outbox: [lefttop.x lefttop.y width height]

savedbox = sprintf('savedbox/%s/%s_y%d_x%d.mat',datasettype,detectionmethod,testdata(1).y,testdata(1).x);
if exist(savedbox, 'file')
    load(savedbox);
else
    switch(detectionmethod)
        case 'NO'
            box = get_whole(testdata);
        case 'DPM_VOC'
            box = run_DPM_VOC(testdata);
            save(savedbox, 'box');
        case 'DPM_INRIA'
            box = run_DPM_INRIA(testdata);
            save(savedbox, 'box');
        case 'RCNN'
            box = run_RCNN(testdata);
            save(savedbox, 'box');
        case 'GT'
            margin = 0.4;
            box = get_gtbox(testdata, margin);
        case 'GT_JITTER'
            margin = 0.1;
            box = get_gtbox_jitter(testdata, margin);
        case 'OURS'
            box = get_boxfromhmap(testdata, datasetname, datasettype);
        otherwise
            error('Check available methods for detection!');
    end
end


for i=1:numel(testdata)
    testdata(i).box = box(i,:);
end

end

%%
function box = get_whole(testdata)
sampleimg = imread(testdata(1).im);
W = size(sampleimg,2);
H = size(sampleimg,1);
box = [];
for i=1:numel(testdata)
    box(i,:) = [1,1,W,H];
end
end

%% DPM_VOC
function box = run_DPM_VOC(testdata)
addpath('~/Downloads/voc-release5/');
startup;
fprintf('compiling the code...');
compile;
fprintf('done.\n\n');
load('/home/namhoon/Downloads/voc-release5/VOC2007/person_grammar_final');
model.class = 'person grammar';
model.vis = @() visualize_person_grammar_model(model, 6);

box = [];
time_total = 0;
for i=1:numel(testdata)
    fprintf('detecting box for image %d ..\n', i);
    tic;
    box(i,:) = test(testdata(i).im, model, -0.6);
    time = toc;
    time_total = time_total + time;
end
fprintf('total processing time (DPM_VOC): %.4f \n', time_total);
fprintf('  avg processing time (DPM_VOC): %.4f \n', time_total/numel(testdata));
end

%% DPM_INRIA
function box = run_DPM_INRIA(testdata)
addpath('~/Downloads/voc-release5/');
startup;
fprintf('compiling the code...');
compile;
fprintf('done.\n\n');
load('/home/namhoon/Downloads/voc-release5/INRIA/inriaperson_final');
model.vis = @() visualizemodel(model, ...
    1:2:length(model.rules{model.start}));

box = [];
time_total = 0;
for i=1:numel(testdata)
    fprintf('detecting box for image %d ..\n', i);
    tic;
    box(i,:) = test(testdata(i).im, model, -0.3);
    time = toc;
    time_total = time_total + time;
end
fprintf('total processing time (DPM_INRIA): %.4f \n', time_total);
fprintf('  avg processing time (DPM_INRIA): %.4f \n', time_total/numel(testdata));
end

%% FIND BOX (DPM)
function box = test(imname, model, thresh)

% load and display image
im = imread(imname);
% clf;
% image(im);
% axis equal; 
% axis on;
% disp('input image');
% disp('press any key to continue'); pause;
% disp('continuing...');

% load and display model
% model.vis();
% disp([cls ' model visualization']);
% disp('press any key to continue'); pause;
% disp('continuing...');

% detect objects
[ds, bs] = imgdetect(im, model, thresh);
%% NL
if isempty(ds)
    box = [1,1,size(im,2),size(im,1)];
    return; 
end
top = nms(ds, 0.5);

% clf;
if model.type == model_types.Grammar
  bs = [ds(:,1:4) bs];
end
% showboxes(im, reduceboxes(model, bs(top,:)));
% disp('detections');
% disp('press any key to continue'); pause;
% disp('continuing...');

if model.type == model_types.MixStar
  % get bounding boxes
  bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
  bbox = clipboxes(im, bbox);
  top = nms(bbox, 0.5);
%   clf;
%   showboxes(im, bbox(top,:));
%   disp('bounding boxes');
%   disp('press any key to continue'); pause;
end

%% final outbox
if model.type == model_types.MixStar
    if numel(top) > 1
        boxsize = bbox(:,3).*bbox(:,4);
        boxsize = boxsize(top);
        [~, idx] = max(boxsize);
        top = top(idx);
    end
    selected = bbox(top,:);
else
    if numel(top) > 1
        boxsize = ds(:,3).*ds(:,4);
        boxsize = boxsize(top);
        [~, idx] = max(boxsize);
        top = top(idx);
    end
    selected = ds(top,:);
end
box = [selected(1) selected(2) selected(3)-selected(1) selected(4)-selected(2)];

end

%% FASTER RCNN
function box = run_RCNN(testdata)

clear mex;
clear is_valid_handle; % to clear init_key
workingdir = '/home/namhoon/Downloads/faster_rcnn-master';
run(fullfile(workingdir, 'startup'));

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;

opts.test_scales            = 600;

%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(workingdir, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers'); %% VGG-16
% model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers'); %% VGG-16
%model_dir                   = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF'); %% ZF
proposal_detection_model    = load_proposal_detection_model(model_dir);

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_proposal.image_means = gpuArray(proposal_detection_model.conf_proposal.image_means);
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end

% caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, 3)*128);
    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
end

%% -------------------- TESTING --------------------
box = [];
running_time = [];
for j = 1:numel(testdata)
    
    im = imread(testdata(j).im);
    
    if opts.use_gpu
        im = gpuArray(im);
    end
    
    % test proposal
    th = tic();
    [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
    t_proposal = toc(th);
    th = tic();
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    t_nms = toc(th);
    
    % test detection
    th = tic();
    if proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
    t_detection = toc(th);
    
%     fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', testdata(j).im, ...
%         size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, t_proposal, t_nms+t_detection);
    running_time(end+1) = t_proposal + t_nms + t_detection;
    
    % visualize
    classes = proposal_detection_model.classes;
    boxes_cell = cell(length(classes), 1);
    thres = 0.6;
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
        
        I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
%     figure(j);
%     showboxes(im, boxes_cell, classes, 'voc');
%     pause(0.1);
    
    %% save box
    if ~isempty(boxes_cell{15}) % person detected
%         assert(size(boxes_cell{15},1) == 1);    % for now, only take care of only one box 
        tmp = boxes_cell{15};        
        if size(tmp,1) > 1
            [val, idx] = sort(tmp(:,5),'descend');
            tmp = tmp(idx(1),:);
        end
        box(j,:) = [tmp(1) tmp(2) tmp(3)-tmp(1) tmp(4)-tmp(2)];
    else
        box(j,:) = [1 1 size(im,2) size(im,1)]; 
    end
end
fprintf('mean time: %.4fs\n', mean(running_time));

caffe.reset_all();
clear mex;

end

function proposal_detection_model = load_proposal_detection_model(model_dir)
ld                          = load(fullfile(model_dir, 'model'));
proposal_detection_model    = ld.proposal_detection_model;
clear ld;

proposal_detection_model.proposal_net_def ...
    = fullfile(model_dir, proposal_detection_model.proposal_net_def);
proposal_detection_model.proposal_net ...
    = fullfile(model_dir, proposal_detection_model.proposal_net);
proposal_detection_model.detection_net_def ...
    = fullfile(model_dir, proposal_detection_model.detection_net_def);
proposal_detection_model.detection_net ...
    = fullfile(model_dir, proposal_detection_model.detection_net);

end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% to speed up nms
if per_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
end
% do nms
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
end
if after_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
end
end

%% GT BOX
function box = get_gtbox(testdata, margin)

sampleimg = imread(testdata(1).im);
W = size(sampleimg,2);
H = size(sampleimg,1);

box = [];
for i=1:numel(testdata)
    
    joints = testdata(i).point;
    joints = joints(1:14,:);
    
    % get rid of occluded joints
    occ = testdata(i).occ;
    occ = occ(1:14);
    joints(find(occ==1),:) = [];
    
    x1 = min(joints(:,1));
    x2 = max(joints(:,1));
    y1 = min(joints(:,2));
    y2 = max(joints(:,2));
    
    w = x2-x1;
    h = y2-y1;
    
    x1 = x1 - w*margin;
    x2 = x2 + w*margin;
    y1 = y1 - h*margin;
    y2 = y2 + h*margin;
    
    x1 = max(1,x1);
    y1 = max(1,y1);
    x2 = min(x2,W);
    y2 = min(y2,H);
    
    box(i,:) = [x1 y1 x2-x1 y2-y1];
end

end

%% GT BOX JITTER
function box = get_gtbox_jitter(testdata, margin)

sampleimg = imread(testdata(1).im);
W = size(sampleimg,2);
H = size(sampleimg,1);

box = [];
for i=1:numel(testdata)
    
    joints = testdata(i).point;
    joints = joints(1:14,:);
    
    % get rid of occluded joints
    occ = testdata(i).occ;
    occ = occ(1:14);
    joints(find(occ==1),:) = [];
    
    x1 = min(joints(:,1));
    x2 = max(joints(:,1));
    y1 = min(joints(:,2));
    y2 = max(joints(:,2));
    
    w = x2-x1;
    h = y2-y1;
    
    center_x = (x1+x2)/2;
    center_y = (y1+y2)/2;
    
%     clf;
%     figure(1); imshow(testdata(i).im); hold on;
%     rectangle('Position', [x1,y1,w,h]);
    
    % jitter
    % translate center
    center_x = center_x + normrnd(0, margin*w);
    center_y = center_y + normrnd(0, margin*h);
    % jitter width and height
    w  = w  + normrnd(0, margin*w);
    h  = h  + normrnd(0, margin*h);
    % jittered box: x1, x2, y1, y2
    x1 = center_x - (w/2);
    x2 = center_x + (w/2);
    y1 = center_y - (h/2);
    y2 = center_y + (h/2);
    
    x1 = max(1,x1);
    y1 = max(1,y1);
    x2 = min(x2,W);
    y2 = min(y2,H);
    
    box(i,:) = [x1 y1 x2-x1 y2-y1];
    
%     rectangle('Position', [x1,y1,w,h]);
%     hold off;
end

end

%% box from heat map
function box = get_boxfromhmap(testdata, datasetname, datasettype)

% directory to heat map predictions
basedir = '~/develop/PoseRegression/save/PR_fcn/option';
if testdata(1).y == 138 && testdata(1).x == 167
    w = 71; 
    h = 102;
    bestmodel = 1;
    date = 't_SunApr1705:37:282016';
elseif testdata(1).y == 160 && testdata(1).x == 260
    w = 76;
    h = 109;
    bestmodel = 5;
    date = 't_TueApr1908:35:242016';
elseif testdata(1).y == 170 && testdata(1).x == 570
    w = 78;
    h = 112;
    bestmodel = 3;
    date = 't_TueApr1922:59:002016';
elseif testdata(1).y == 262 && testdata(1).x == 544
    w = 98;
    h = 141;
    bestmodel = 6;
    date = 't_WedApr2008:26:562016';
elseif testdata(1).y == 130 && testdata(1).x == 460
    w = 69;
    h = 99;
    bestmodel = 6;
    date = 't_ThuApr2104:24:462016';
elseif testdata(1).y == 235 && testdata(1).x == 325
    w = 93;
    h = 133;
    bestmodel = 4;
    date = 't_SunMay107:29:372016';
elseif testdata(1).y == 169 && testdata(1).x == 92
    w = 78;
    h = 112;
    bestmodel = 3;
    date = 't_MonApr2504:23:352016';
elseif testdata(1).y == 91  && testdata(1).x == 354
    w = 61;
    h = 87;
    bestmodel = 9;
    date = 't_MonApr2522:31:402016';
elseif testdata(1).y == 230 && testdata(1).x == 438
    w = 91;
    h = 131;
    bestmodel = 5;
    date = 't_WedApr2709:26:522016';
elseif testdata(1).y == 105 && testdata(1).x == 245
    w = 64;
    h = 91;
    bestmodel = 8;
    date = 't_FriApr2917:18:122016';
elseif testdata(1).y == 240 && testdata(1).x == 150
    w = 68;
    h = 97;
    bestmodel = 20;
    date = 't_ThuMay509:29:002016';
elseif testdata(1).y == 270 && testdata(1).x == 550
    w = 76;
    h = 109;
    bestmodel = 22;
    date = 't_FriMay609:55:552016';
elseif testdata(1).y == 250 && testdata(1).x == 340
    w = 71;
    h = 101;
    bestmodel = 28;
    date = 't_MonMay919:51:352016';
elseif testdata(1).y == 420 && testdata(1).x == 130
    w = 93;
    h = 132;
    bestmodel = 10;
    date = 't_SatMay702:56:182016';
else
    error('No available anchor location!');
end
dirtopred = fullfile(basedir, sprintf('%s/results/%s/model%d',date,datasettype,bestmodel));

% load gt tight box
if strcmp(datasetname, 'towncenter')
    step = 1;
elseif strcmp(datasetname, 'pet2006')
    step = 2;
else
    error('invalid datasetname!');
end
tightbox = dlmread(sprintf('~/develop/PoseRegression/mat/tightbox/tightboxes_%s_step%d.txt',datasetname,step));


% get prediction box
box = [];
for i=1:numel(testdata)
    % load jsc
    if strcmp(datasettype, 'rTest')    
        framenum = testdata(i).im(end-7:end-4);
        jsc = fullfile(dirtopred, sprintf('jsc_pred_frm%s.mat',framenum));
    elseif strcmp(datasettype, 'sTest')
        jsc = fullfile(dirtopred, sprintf('jsc_pred_frm%04d.mat',i));
    else
        error('invalid datasettype!');
    end
    load(jsc);
    
    % get center channel
    x = permute(x, [3 4 2 1]);
    cen = x(:,:,29);
    % find peak of center channel
    [max_val,idx] = max(cen(:));
    [center.y,center.x] = ind2sub(size(cen), idx);
    % translate the center to the absolute coordinate w.r.t. the frame
    % l r t b
    % ctot
    ctot = (h - h/3.5*2)/2 + h/3.5*2;
    ctol = w/2;
    center.y_abs = center.y + (testdata(1).y - ctot);
    center.x_abs = center.x + (testdata(1).x - ctol);

    % find the closest location from tightbox
    dist = tightbox(:,1:2) - repmat([center.x_abs center.y_abs], [size(tightbox,1),1]);
    dist = sqrt(sum(dist(:,1).^2 + dist(:,2).^2, 2));
    [min_val,idx] = min(dist);
    
    % w and h
    w = tightbox(idx,3);
    h = tightbox(idx,4);
    
    % box 
    box(i,:) = [center.x-w/2 center.y-h/2 w h];    
%     figure(1); imshow(testdata(i).im); hold on;
%     rectangle('Position', box(i,:));
%     hold off;
    
end


end





