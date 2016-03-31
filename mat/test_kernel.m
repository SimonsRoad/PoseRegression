clc;clear; close all;


sigma = 1;

%% test create 2d gaussian
[X,Y] = meshgrid(1:16,1:32);
keypoint.x = 8;
keypoint.y = 6;

kernel1 = (1/(sigma*sqrt(2*pi))) * exp(-((X-keypoint.x).^2 + (Y-keypoint.y).^2) / (2*sigma^2));
kernel1 = kernel1/max(kernel1(:));
figure(1); imagesc(kernel1); axis image;


%% generate using fspecial
h = fspecial('gaussian', [32,16], sigma);
h = h/max(h(:));
figure(2);imagesc(h); axis image;
