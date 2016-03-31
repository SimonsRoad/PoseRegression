% crop_location.m
% Namhoon Lee (namhoonl@andrew.cmu.edu)
% 
clc; clear;

%% inputs to function
fpath_in  = '~/Downloads/data_towncenter/frames/';
fpath_out = '~/Downloads/data_towncenter/frames_out/';
fnum      = 111;        % 54, 105, 111, 214, 256, 300
fname_in  = fullfile(fpath_in,  sprintf('frame%06d.jpg', fnum));
fname_out = fullfile(fpath_out, sprintf('frame%06d.jpg', fnum));

crop.w = 47;
crop.h = 87;


%% main function: crop_location
%- draw rectangle
pos.x = 224; pos.y = 74;
pos.rec = [pos.x, pos.y, crop.w, crop.h];
img = imread(fname_in);
figure; imshow(img); hold on;
rectangle('Position', pos.rec); 

%- crop
img_crop = img(pos.y:pos.y+crop.h, pos.x:pos.x+crop.w, :);
%figure; imshow(img_crop);

%- save
imwrite(img_crop, fname_out);