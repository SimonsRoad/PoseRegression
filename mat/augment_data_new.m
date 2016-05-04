% augment_data_new.m
% 
% perform data augmentation and save those.

clear; clc; close all;


%% load existing data
% data directories
y = 250;
x = 340;
path_data = sprintf('~/develop/PoseRegression/data/rendout/anc_y%d_x%d', y, x);
path_pos  = fullfile(path_data, 'pos');
path_seg  = fullfile(path_data, 'seg');
path_jsc  = fullfile(path_data, 'jsc');
if ~exist(path_jsc, 'dir'), mkdir(path_jsc); end;

% images
imgtype = '*.jpg';
images  = dir(fullfile(path_pos, imgtype));

data = [];
for i=1:numel(images)
    data(i).pos = fullfile(path_pos, images(i).name);
    data(i).seg = fullfile(path_seg, strrep(images(i).name,'pos0000', 'seg0000'));
end
[h,w,~] = size(imread(data(1).pos));

% label
for i = 1:numel(data)
    tmp = data(i).pos;
    tmp = strrep(tmp, '/pos/', '/j27/');
    tmp = strrep(tmp, 'pos0000.jpg', 'joints.txt');
    data(i).j27 = dlmread(tmp);
end

% images - resized
factor_resize = 0.5;
for i = 1:5
    im = imread(data(i).pos);
    im = imresize(im, factor_resize);
    fname = strrep(data(i).pos, '/pos/', '/pos_half/');
    data(i).pos_half = fname;
    imwrite(im, fname);
end
[h,w,~] = size(imread(data(1).pos_half));


%% generate jsc [joint, segmentation, center] label
nJoints = 27;
for i = 1:numel(data)
    fprintf('processing %d..\n', i);

    % joint
    j27 = data(i).j27;
    j27 = j27 .* repmat([w, h], [nJoints,1]);
    j27_round = round(j27);
    
    % hmap
    hmap = single(zeros(h,w,nJoints));
    for j = 1:nJoints
        if j27_round(j,2) > h || j27_round(j,1) > w || sum(j27_round(j,:)<=0)
            continue;
        else
            hmap(j27_round(j,2), j27_round(j,1), j) = 1;
            hmap(:,:,j) = imgaussfilt(hmap(:,:,j), 3);       % gaussian
%             hmap(:,:,j) = imgaussfilt(hmap(:,:,j), 1.5);       % gaussian half sigma
            hmap(:,:,j) = hmap(:,:,j)/max(max(hmap(:,:,j)));  % normalize approach1
        end
    end
    
    % seg
    seg = imread(data(i).seg);
    seg = imresize(seg, factor_resize);
    seg = im2single(rgb2gray(seg));

    % cen
    midpoint = round( mean([j27(6,:);j27(12,:)]) );
    cen = single(zeros(h,w));
    if midpoint(1) > w || midpoint(2) > h || sum(midpoint<=0)
    else
        cen(midpoint(2), midpoint(1)) = 1;
        cen = imgaussfilt(cen, 5); 
%         cen = imgaussfilt(cen, 2.5); % gaussian half sigma
        cen = single(cen);
        cen = cen/max(cen(:));
    end
    
    %% concatenated label: JSC
    jsc = single([]);
    jsc(:,:,1:27) = hmap;
    jsc(:,:,28)   = seg;
    jsc(:,:,29)   = cen;
    jsc = permute(jsc,[3, 1, 2]);         % following Torch standard
    
    % sanity check: jsc
    if 0
        visualize_jsc(imread(data(i).pos_half), permute(jsc,[2, 3, 1]), permute(jsc,[2, 3, 1]), nJoints);
    end
    
    % save jsc
    fname_jsc = strrep(data(i).pos, '/pos/', '/jsc/');
    fname_jsc = strrep(fname_jsc, 'pos0000.jpg', 'jsc.mat');
    save(fname_jsc, 'jsc');
        
end


%% generate negative samples
if 0
path_neg            = fullfile(path_data, 'neg');
path_neg_aug        = fullfile(path_data_aug, 'neg');
path_jsdc_neg_aug   = fullfile(path_data_aug, 'jsdc_neg');
if ~exist(path_neg_aug,         'dir'), mkdir(path_neg_aug); end;
if ~exist(path_jsdc_neg_aug,    'dir'), mkdir(path_jsdc_neg_aug); end;

% generate!
negatives= dir(fullfile(path_neg, imgtype));
assert(numel(negatives)==5000);
jsdc = single(zeros(30, 128, 64));
for i=1:numel(negatives)
    imgname = fullfile(path_neg, negatives(i).name);
    
    % process image (resize and crop)
    img = imread(imgname);
    bw_outer = 90;
    bh_outer = bw_outer * 2;
    bw_crop  = 64;
    bh_crop  = bw_crop * 2;
    img_new = imresize(img, [bh_outer, bw_outer]);
    img_new = img_new(...       % crop
        floor((bh_outer-bh_crop)/2)+1:floor((bh_outer-bh_crop)/2)+bh_crop, ...
        floor((bw_outer-bw_crop)/2)+1:floor((bw_outer-bw_crop)/2)+bw_crop, :); 
    assert(size(img_new,1)==128 & size(img_new,2)==64);
    
    % save (negative) image
    fname_neg = fullfile(path_neg_aug, negatives(i).name);
    imwrite(img_new, fname_neg);
    
    % save (negative) label: Empty channels
    fname_neg_jsdc = fullfile(path_jsdc_neg_aug, sprintf('%d.mat',i));
    save(fname_neg_jsdc, 'jsdc');
end
end

%% sanity check: visualize joints on images
if 0
images_aug = dir(fullfile(path_pos_aug, imgtype));

% images
data_aug = [];
for i=1:100 %numel(images_aug)
    data_aug(i).pos = fullfile(path_pos_aug, images_aug(i).name);
end

% label
for i = 1:numel(data_aug)
    tmp = data_aug(i).pos;
    tmp = strrep(tmp, 'pos', 'jsdc');
    tmp = strrep(tmp, 'im', 'jsdc');
    tmp = strrep(tmp, 'jpg', 'mat');
    data_aug(i).jsdc = load(tmp);
end

% visualize one randomly selected sample
randsample = randi([1, numel(data_aug)], 1);
img_sample = imread(data_aug(randsample).pos);
jsdc_sample= permute(data_aug(randsample).jsdc.jsdc, [2, 3, 1]);
hmap = jsdc_sample(:,:,1:27);
plot_joints_hmap(img_sample, hmap);

figure;
subplot(1,3,1); imshow(jsdc_sample(:,:,28));
subplot(1,3,2); imshow(jsdc_sample(:,:,29));
subplot(1,3,3); imshow(jsdc_sample(:,:,30));
end


