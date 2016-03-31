% augment_data.m
% 
% perform data augmentation and save those.

clear; clc; close all;


%% load existing data
% data directories
path_data = '~/develop/PoseRegression/data/rendout/tmp_y144_x256';
path_pos  = fullfile(path_data, 'pos');
path_seg  = fullfile(path_data, 'seg');
path_dep  = fullfile(path_data, 'dep');

% create save directories
path_data_aug       = [path_data, '_aug'];
path_pos_aug        = fullfile(path_data_aug, 'pos');
path_jsdc_aug       = fullfile(path_data_aug, 'jsdc');
if ~exist(path_data_aug,    'dir'), mkdir(path_data_aug); end;
if ~exist(path_pos_aug,     'dir'), mkdir(path_pos_aug); end;
if ~exist(path_jsdc_aug,    'dir'), mkdir(path_jsdc_aug); end;

% images
imgtype = '*.jpg';
images  = dir(fullfile(path_pos, imgtype));

data = [];
for i=1:numel(images)
    data(i).pos = fullfile(path_pos, images(i).name);
    data(i).seg = fullfile(path_seg, images(i).name);
    data(i).dep = fullfile(path_dep, images(i).name);
end

% label
for i = 1:numel(data)
    tmp = data(i).pos;
    pid     = tmp(strfind(tmp, 'Ped_id')+6:strfind(tmp, 'Ped_id')+10);
    pose    = tmp(strfind(tmp, 'pose')  +4:strfind(tmp, 'pose')  +8);
    rot     = tmp(strfind(tmp, 'rot')   +3:strfind(tmp, 'rot')   +5);
    hscale  = tmp(strfind(tmp, 'hscale')+6:strfind(tmp, 'hscale')+6);
    data(i).j27 = dlmread(fullfile(path_data, ['j27/2D_Ped_id',pid,'_pose',pose,'_rot',rot, '_hscale', hscale, '.txt']));
end



%% augmentation
% instead of random crop, crop all possible conditions.
for i = 1:numel(data)
    fprintf('processing %d..\n', i);
    
    bw_outer = 90;
    bh_outer = bw_outer * 2;
    bw_crop  = 64;
    bh_crop  = bw_crop * 2;
    
    pos = imread(data(i).pos);
    seg = imread(data(i).seg);
    dep = imread(data(i).dep);
    cen = fspecial('gaussian', [bh_outer, bw_outer], 5);    
    
    
    %% transformations! seg. dep, cen, j27
    % 1. convert from 3 channels to 1 channel
    % 2. convert from double to single precision
    % 3. create joint heatmap
    seg = im2single(rgb2gray(seg));
    dep = im2single(rgb2gray(dep));
    cen = single(cen);

    w_ori = size(pos,2);
    h_ori = size(pos,1);
    
    label_x_min = min(data(i).j27(:,1));
    label_x_max = max(data(i).j27(:,1));
    label_y_min = min(data(i).j27(:,2));
    label_y_max = max(data(i).j27(:,2));
    
    lt_tight_x = label_x_min * bw_outer;
    lt_tight_y = label_y_min * bh_outer;
    rb_tight_x = label_x_max * bw_outer;
    rb_tight_y = label_y_max * bh_outer;
    bw_tight = rb_tight_x - lt_tight_x;
    bh_tight = rb_tight_y - lt_tight_y;
    
    d_from_tight_to_centerbox_x = (bw_crop-bw_tight)/2;
    d_from_tight_to_centerbox_y = (bh_crop-bh_tight)/2;
    lt_crop_x_center = lt_tight_x - d_from_tight_to_centerbox_x;
    lt_crop_y_center = lt_tight_y - d_from_tight_to_centerbox_y;
    
    lt_crop_x_min = lt_crop_x_center - 4;
    lt_crop_y_min = lt_crop_y_center - 4;
    lt_crop_x_max = lt_crop_x_center + 4;
    lt_crop_y_max = lt_crop_y_center + 4;
    
    if floor(lt_crop_x_min) <= 0
        lt_crop_x_min = 1;
    end
    if ceil(lt_crop_x_max)+bw_crop >= bw_outer
        lt_crop_x_max = bw_outer-bw_crop-1;
    end
    if floor(lt_crop_y_min) <= 0
        lt_crop_y_min = 1;
    end
    if ceil(lt_crop_y_max)+bh_crop >= bh_outer
        lt_crop_y_max = bh_outer-bh_crop-1;
    end
    
    
    for dx = round(lt_crop_x_min):2:round(lt_crop_x_max)
        for dy = round(lt_crop_y_min):2:round(lt_crop_y_max)
            %% augmentations..
            % resize
            pos_new = imresize(pos, [bh_outer, bw_outer]);
            seg_new = imresize(seg, [bh_outer, bw_outer]);
            dep_new = imresize(dep, [bh_outer, bw_outer]);
            
            % crop images (pos, seg, dep)
            pos_new = pos_new(dy+1:dy+bh_crop, dx+1:dx+bw_crop, :);
            seg_new = seg_new(dy+1:dy+bh_crop, dx+1:dx+bw_crop, :);
            dep_new = dep_new(dy+1:dy+bh_crop, dx+1:dx+bw_crop, :);
            cen_new = cen(dy+1:dy+bh_crop, dx+1:dx+bw_crop);
            
            % normalize 
            seg_new = seg_new/max(seg_new(:));
            dep_new = dep_new/max(dep_new(:));
            cen_new = cen_new/max(cen_new(:));

            assert(size(pos_new,1) == bh_crop & size(pos_new,2) == bw_crop);
            
            % label transform
            joints = data(i).j27;
            joints = joints .* repmat([bw_outer, bh_outer], [27,1]);
            joints = joints -  repmat([dx, dy],             [27,1]);
            if min(joints(:,1)) <= 0 || max(joints(:,1) >= bw_outer)
                assert(false);
            end
            
            % j27 heatmap
            j27 = round(joints);
            hmap = single(zeros(128,64,size(j27,1)));
            for j = 1:size(j27,1)
                hmap(j27(j,2), j27(j,1), j) = 1;
%                 hmap(:,:,j) = i                                                                                                                                           mgaussfilt(hmap(:,:,j), 2);       % gaussian
                hmap(:,:,j) = imgaussfilt(hmap(:,:,j), 3);       % gaussian
                hmap(:,:,j) = hmap(:,:,j)/max(max(hmap(:,:,j))); % normalize
            end
            
            %% visualize and check
            if 0
                plot_joints_hmap(pos_new, hmap);
            end
    
            %% create concatenated label: JSDC
%             jsdc = single([]); 
%             jsdc(:,:,1:27) = hmap;
%             jsdc(:,:,28)   = seg_new;
%             jsdc(:,:,29)   = dep_new;
%             jsdc(:,:,30)   = cen_new;
%             jsdc = permute(jsdc,[3, 1, 2]);         % following Torch standard

            
            %% save
            % image: pos
            fname_pos = fullfile(path_pos_aug, sprintf('im%05d_dx%03d_dy%03d.jpg',i,dx,dy));
            imwrite(pos_new, fname_pos);
            
            % label: jsdc
%             fname_jsdc = fullfile(path_jsdc_aug, sprintf('jsdc%05d_dx%03d_dy%03d.mat',i,dx,dy));
%             save(fname_jsdc, 'jsdc');
        end
    end
    
end


%% generate negative samples
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


