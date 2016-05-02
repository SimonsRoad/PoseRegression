function testdata = run_detection(testdata, detectionmethod)
% outbox: [lefttop.x lefttop.y width height]

switch(detectionmethod)
    case 'NO'
        box = get_whole(testdata);
    case 'DPM_VOC'
        box = run_DPM_VOC(testdata);
    case 'DPM_INRIA'
        box = run_DPM_INRIA(testdata);
    case 'GT'
        margin = 0.4;
        box = get_gtbox(testdata, margin);
    otherwise
        error('Check available methods for detection!');
end

for i=1:numel(testdata)
    testdata(i).box = box(i,:);
end

end

function box = get_whole(testdata)
sampleimg = imread(testdata(1).im);
W = size(sampleimg,2);
H = size(sampleimg,1);
box = [];
for i=1:numel(testdata)
    box(i,:) = [1,1,W,H];
end
end

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