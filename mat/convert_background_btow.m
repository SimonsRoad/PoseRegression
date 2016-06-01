% convert_background_btow.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)

clc; clear; close all;

% img list
basedir = '~/3Dmodels/seg';
data = dir(basedir); 

% convert background from black to white
for i = 1:numel(data)
    if ~data(i).isdir
        img = imread(fullfile(basedir, data(i).name));
        img_bw = im2bw(img);
        idx_black = find(img_bw==0);
        idx_white = find(img_bw==1);
        
        % new image
        img_r = zeros(size(img_bw));
        img_g = zeros(size(img_bw));
        img_b = zeros(size(img_bw));
        img_r(idx_black) = 255;
        img_g(idx_black) = 255;
        img_g(idx_white) = 255;
        img_b(idx_black) = 255;
        img_new(:,:,1) = img_r;
        img_new(:,:,2) = img_g;
        img_new(:,:,3) = img_b;
        
        figure(1); 
        subplot(1,2,1); imshow(img);
        subplot(1,2,2); imshow(img_new);
    end
end