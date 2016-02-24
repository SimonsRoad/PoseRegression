% get_joints.m
% Namhoon Lee, RI, Carnegie Mellon University.
% namhoonl@andrew.cmu.edu
% extract the list of joints text files

clear;clc;

projdir = '~/develop/PoseRegression';
imglist = fullfile(projdir, 'data/lists/onlytest.txt');
fid = fopen(imglist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    allimg(cnt).im = tline;
    tline = fgetl(fid);
end
fclose(fid);
[h,w,~] = size(imread(allimg(1).im));

tmp = [];
for i = 1:numel(allimg)
    tmp = allimg(i).im;
    pid  = tmp(strfind(tmp, 'Ped_id')+6:strfind(tmp, 'Ped_id')+10);
    pose = tmp(strfind(tmp, 'pose')+4:strfind(tmp, 'pose')+8);
    rot  = tmp(strfind(tmp, 'rot')+3:strfind(tmp, 'rot')+5);
    tmp = dlmread([projdir, '/data/rendout/joints_norm/2D/loc045001/Ped_id',pid,'_pose',pose,'_rot',rot,'.txt']);
    tmp = tmp .* repmat([w h], 14, 1);
    tmp = tmp([14 13 12 6 7 8 11 10 9 3 4 5 2 1], :);
    allimg(i).point = tmp;
end

% check joints. temporary.
tmpimg = imread(allimg(1).im);
tmpjoint = allimg(1).point;
figure(1);imshow(tmpimg); hold on;
for i=1:size(tmpjoint,1)
    plot(tmpjoint(i,1), tmpjoint(i,2), 'r*');
end