%- visualize_groundtruth.m
% check if data and labels match
% the file name should be 'thisisatestset.mat' or equivalent..

clc;clear;

%- load saved file
fpath = '../save/PR_multi/option,t=PR_multi/t_SatFeb2717:04:292016';
fname = fullfile(fpath, 'testdata.mat');
load(fname);
data = permute(data,[3 4 2 1]);
resz = 4;
nJoints = 14;

img = [];
for iSmp = 1:size(data,4) 
    img = data(:,:,:,iSmp);
    img = imresize(img, resz);
    
    joints_gt = reshape(label(iSmp, :), [2,nJoints]);
    joints_gt = joints_gt'.*repmat(resz*[64,128],[nJoints,1]);
    disp([joints_gt]);
    
    clf; figure(1); set(gcf, 'Position', [600,800,600,1200]);
%     img = insertText(img, joints_pred, jName, 'FontSize', 8);
    imshow(img); hold on;
    plot(joints_gt(:,1), joints_gt(:,2), 'b*', 'MarkerSize', 5);
    draw_sticks_joints(joints_gt, nJoints, 'b');
    drawnow;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            now;
    

end

