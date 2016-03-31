% create_27ch_joint.m
% This gives the 27 channel heat map of joints

clear; clc; close all;

% load joint
jointpath = '~/develop/PoseRegression/data/rendout/tmp_y144_x256_aug/j27'; 
imgpath   = '~/develop/PoseRegression/data/rendout/tmp_y144_x256_aug/pos';
jointtype = '*.txt';
joints    = dir(fullfile(jointpath, jointtype));

% save joint
jointpath_save = [jointpath, '_hmap'];
if ~exist(jointpath_save ,'dir'), mkdir(jointpath_save), end

for i = 1:numel(joints)
    fprintf('processing %d th.. \n', i);
    
    % read joints
    j27 = dlmread(fullfile(jointpath, joints(i).name));
    j27 = round(j27);
    hmap = zeros(128,64,size(j27,1));
    for j = 1:size(j27,1)
        hmap(j27(j,2), j27(j,1), j) = 1;                
        hmap(:,:,j) = imgaussfilt(hmap(:,:,j), 2);       % gaussian
        hmap(:,:,j) = hmap(:,:,j)/max(max(hmap(:,:,j))); % normalize
    end

    % visualize
    if 0
        img = imread(fullfile(imgpath, strrep(strrep(joints(i).name, 'txt', 'jpg'), 'joint', 'im')));
        plot_joints_hmap(img, hmap);
    end
    
    
    % save
    save(fullfile(jointpath_save, strrep(joints(i).name, 'txt', 'mat')), 'hmap');
end

