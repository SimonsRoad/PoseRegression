% create_jsdc_for_real.m
% 
% This code assumes that you already have 
% 1) a list of images in .txt
% 2) labels that matches to the images in .txt
% Then, the code generates jsdc labels.
% 
clear; clc; close all;


test = [];

% images (approach2 - essentially should be gone)
imglist = '~/develop/towncenter/data/frames_y144_x256_sel/list.txt';
fid = fopen(imglist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    test(cnt).im = tline;
    tline = fgetl(fid);
end
fclose(fid);

% load joints
load('~/develop/towncenter/data/frames_y144_x256_sel/labeled_data_NL.mat')
for i = 1:numel(test)
    joints = label_joint_pos_rel_NL{1,i};
    joints = joints .* repmat(bbox_size(i,end:-1:1), 14, 1);
    test(i).point  = joints;
end

% check one random sample
if 0
    randsample = randi([1, numel(test)], 1);
    plot_joints(imread(test(randsample).im), test(randsample).point);
end

% create hmap, jsdc and save jsdc
for i = 1:numel(test)
    j14 = round(test(i).point);
    hmap = single(zeros(128,64,size(j14,1)));
    for j = 1:size(j14,1)
        hmap(j14(j,2), j14(j,1), j) = 1;
        %                 hmap(:,:,j) = i                                                                                                                                           mgaussfilt(hmap(:,:,j), 2);       % gaussian
        hmap(:,:,j) = imgaussfilt(hmap(:,:,j), 3);       % gaussian
        hmap(:,:,j) = hmap(:,:,j)/max(max(hmap(:,:,j))); % normalize
    end
    
    % sanity check
    if 0
        plot_joints_hmap(imread(test(i).im), hmap);
    end
    
    % create jsdc
    jsdc = single([]);
    jsdc(:,:,1:14) = hmap;
    jsdc(:,:,15:27)= single(zeros(128, 64, 13));
    jsdc(:,:,28)   = single(zeros(128, 64));
    jsdc(:,:,29)   = single(zeros(128, 64));
    jsdc(:,:,30)   = single(zeros(128, 64));
    jsdc = permute(jsdc,[3, 1, 2]);         % following Torch standard

    % save
    lastslash = max(strfind(test(i).im, '/'));
    savedir = strrep(test(i).im(1:lastslash-1), 'pos', 'jsdc');
    if ~exist(savedir, 'dir'), mkdir(savedir); end;
    fname_jsdc = strrep(test(i).im, 'pos', 'jsdc');
    fname_jsdc = strrep(fname_jsdc, 'jpg', 'mat');
    save(fname_jsdc, 'jsdc');
end
