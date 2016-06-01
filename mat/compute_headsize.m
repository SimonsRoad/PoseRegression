% compute_headsize.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
% read joints file, get the head size and then take the average
%
clc; clear; close all;

y = 999;
x = 999;
w = 78; h = 112;

basedir = sprintf('~/develop/PoseRegression/data/rendout/anc_y%03d_x%d/j27', y, x);
filelist = dir(basedir);

headsz_all = [];
for i=1:numel(filelist)
    fprintf('processing %d .. \n', i);
    if ~filelist(i).isdir
        jnt = dlmread(fullfile(basedir, filelist(i).name));
        head = jnt(1:2,:).*repmat([w h], [2,1]);
        headsz = sqrt((head(1,1)-head(2,1))^2+(head(1,2)-head(2,2))^2);
        headsz_all = [headsz_all; headsz];
    end
    if i>50000
        break;
    end
end

fprintf('mean head size: %.2f \n', mean(headsz_all));
