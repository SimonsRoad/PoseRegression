function eval_detection_precisionrecall()
close all; clear; 

%% add path
addpath('detection');
addpath('~/Downloads/convolutional-pose-machines-release/testing/');


%% Load test data

datasetname = 'pet2006';             % 1) towncenter, 2)pet2006
datasettype = 'rTest';

RCNNInitFlag = 0;
total_gt_rects = 0;
all_labels_total = [];
all_scores_total = [];
for selLoc = 1:4
    
    if strcmp(datasetname, 'towncenter')
        YX = [...
            138 167;
            160 260;
            170 570;
            262 544;
            130 460;
            235 325;
            169 92;
            91 354;
            230 438;
            105 245;
            999 999];
    elseif strcmp(datasetname, 'pet2006')
        YX = [240 150;
            270 550;
            250 340;
            420 130];
    else
        error('invalid datasetname!');
    end
    y = YX(selLoc,1);
    x = YX(selLoc,2);
    
    testdata = load_dataset(x,y, datasetname, datasettype);
    testdata = load_predictions(testdata, datasettype, 'no');
    
    
    %% Run Detection
    % OURS, DPM_INRIA, RCNN
    detectionmethod = 'RCNN';
    
    fname = sprintf('detectionscores/%s_%s/%s_%d.mat',datasetname,datasettype,detectionmethod,selLoc);
    if exist(fname)
        load(fname);
    else
        switch(detectionmethod)
            case 'OURS'
                box = feval(['RUN_', detectionmethod], testdata, datasetname);
            case 'DPM_INRIA'
                box = feval(['RUN_', detectionmethod], testdata);
            case 'DPM_VOC'
                box = feval(['RUN_', detectionmethod], testdata);
            case 'RCNN'
                [box,RCNNInitFlag] = feval(['RUN_', detectionmethod], testdata, RCNNInitFlag);
            otherwise
                error('invalid detection method!');
        end
        
        
        %% evaluate detection
        gtbox = get_gtbox(testdata, 0.1);
        
        total_gt_rects = total_gt_rects + size(gtbox,1);
        scores = cell(numel(testdata),1);
        labels = cell(numel(testdata),1);
        for i=1:numel(testdata)
            gt_rects = gtbox(i,:);
            rects = box{i};
            
%             [~,idx] = max(rects(:,5));
%             figure(1); imshow(testdata(i).im); hold on;
%             rectangle('Position', gt_rects, 'EdgeColor', 'g');
%             rectangle('Position', rects(idx,1:4), 'EdgeColor', 'r');
%             hold off;
%             figure(2); imagesc(testdata(i).jsc(:,:,29)); axis image;
            
            [labels{i}, scores{i}] = evaluate_image(gt_rects, rects);
        end
        all_labels = cat(1, labels{:});
        all_scores = cat(1, scores{:});
        
        save(fname, 'all_labels', 'all_scores');
    end
    
    idx = all_scores > -1.3;

    [precision, recall, ap, sorted_scores] = ...
        precision_recall(all_labels(idx==1), all_scores(idx==1), numel(testdata), 0);
    
    disp(['AP: ' num2str(ap)])
    
    all_labels_total = [all_labels_total; all_labels(idx==1)];
    all_scores_total = [all_scores_total; all_scores(idx==1)];

end

if strcmp(datasetname, 'towncenter')
    if strcmp(datasettype, 'rTest')
        total_gt_rects = 358;
    elseif strcmp(datasettype, 'sTest')
        total_gt_rects = 1000;
    end
elseif strcmp(datasetname, 'pet2006')
    if strcmp(datasettype, 'rTest')
        total_gt_rects = 1079;
    elseif strcmp(datasettype, 'sTest')
        total_gt_rects = 400;
    end
end


%compute PR curve and AP; a figure will be shown if show_plots is true
[precision, recall, ap, sorted_scores] = ...
    precision_recall(all_labels_total, all_scores_total, total_gt_rects, true);

disp(['avg AP: ' num2str(ap)])

% save precision and recall for later use
fname = sprintf('detectionscores/%s_%s/PR_%s.mat',datasetname,datasettype,detectionmethod);
save(fname, 'precision', 'recall');

end


%%
function box = RUN_OURS(testdata, datasetname)

% load gt tight box
if strcmp(datasetname, 'towncenter')
    step = 1;
elseif strcmp(datasetname, 'pet2006')
    step = 2;
else
    error('invalid datasetname!');
end
tightbox = dlmread(sprintf('~/develop/PoseRegression/mat/tightbox/tightboxes_%s_step%d.txt',datasetname,step));

% detection info
dinfo = load_detector_info(testdata(1).y, testdata(1).x, 'no');
w = dinfo.w;
h = dinfo.h;

box = cell(numel(testdata),1);

for i=1:numel(testdata)
    
    %% settings
    suppressed_scale = 0.8;
    max_peaks = 80;
    threshold = -1.0;
   
    %% get detection_size
    % get center channel
    cen = testdata(i).jsc(:,:,29);
    % find peak of center channel
    [max_val,idx] = max(cen(:));
    [center.y,center.x] = ind2sub(size(cen), idx);
    % translate the center to the absolute coordinate w.r.t. the frame
    % l r t b
    % ctotall_scores
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
    
    detection_size = [h w];
    response = testdata(i).jsc(:,:,29);
    
    rects = get_rects(response, max_peaks, threshold, detection_size, suppressed_scale);
    rects = post_process_rects(rects, 0.5);
    
    box{i} = rects;
end

end


%%
function box = RUN_DPM_INRIA(testdata)
addpath('~/Downloads/voc-release5/');
startup;
fprintf('compiling the code...');
% compile;
fprintf('done.\n\n');
load('/home/namhoon/Downloads/voc-release5/INRIA/inriaperson_final');
model.vis = @() visualizemodel(model, ...
    1:2:length(model.rules{model.start}));

box = cell(numel(testdata),1);
for i=1:numel(testdata)
    fprintf('detecting box for image %d ..\n', i);
    box{i} = test(testdata(i).im, model, -100);
end

end


%%
function box = RUN_DPM_VOC(testdata)
addpath('~/Downloads/voc-release5/');
startup;
fprintf('compiling the code...');
% compile;
fprintf('done.\n\n');
load('/home/namhoon/Downloads/voc-release5/VOC2010/person_grammar_final');
model.class = 'person grammar';
model.vis = @() visualize_person_grammar_model(model, 6);

box = cell(numel(testdata),1);
for i=1:numel(testdata)
    fprintf('detecting box for image %d ..\n', i);
    box{i} = test(testdata(i).im, model, -100);
end
end


%%
function box = test(imname, model, thresh)

im = imread(imname);


% detect objects
[ds, bs] = imgdetect(im, model, thresh);
if isempty(ds)
    box = [1,1,size(im,2),size(im,1), thresh];
    return; 
end
top = nms(ds, 0.5);

% clf;
if model.type == model_types.Grammar
  bs = [ds(:,1:4) bs];
end
% showboxes(im, reduceboxes(model, bs(top,:)));

if model.type == model_types.MixStar
  % get bounding boxes
  bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
  bbox = clipboxes(im, bbox);
  top = nms(bbox, 0.5);
%   clf;
%   showboxes(im, bbox(top,:));
end

%% final outbox
if model.type == model_types.MixStar
    sel = bbox(top,:);
else
    sel = ds(top,:);
end
box = [sel(:,1) sel(:,2) sel(:,3)-sel(:,1) sel(:,4)-sel(:,2), sel(:,5)];

end


%%
function [box,RCNNInitFlag] = RUN_RCNN(testdata, RCNNInitFlag)

clear mex;
clear is_valid_handle; % to clear init_key
workingdir = '/home/namhoon/Downloads/faster_rcnn-master';
run(fullfile(workingdir, 'startup'));

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
if ~RCNNInitFlag
    active_caffe_mex(opts.gpu_id, opts.caffe_version);
    RCNNInitFlag = 1;
end
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
box = cell(numel(testdata),1);
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
%     thres = 0.6;
    thres = -100;
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
        sel = boxes_cell{15};
        box{j} = [sel(:,1) sel(:,2) sel(:,3)-sel(:,1) sel(:,4)-sel(:,2), sel(:,5)];
    else
        box{j} = [1,1,size(im,2),size(im,1), thres];
    end
end
fprintf('mean time: %.4fs\n', mean(running_time));

% caffe.reset_all();

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

%%
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