% sanitycheck_dataset.m

clear; close all; clc;


pathtotxt = '~/develop/PoseRegression/data/rendout/tmp_y144_x256_aug/lists';
dataset = [];


% load image
fImage = fullfile(pathtotxt, 'img_8192.txt');
fid = fopen(fImage);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    if cnt > 6000, break; end;
    dataset(cnt).img = tline;
    tline = fgetl(fid);
end
fclose(fid);

% load label
fLabel = fullfile(pathtotxt, 'jsdc_8192.txt');
fid = fopen(fLabel);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    if cnt > 6000, break; end;
    tmp = load(tline);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    dataset(cnt).jsdc = tmp.jsdc;
    tline = fgetl(fid);                                                                                                                                                                                                                                                                                                                                                                                                                             
end
fclose(fid);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                
% visualize 
randidx = randperm(numel(dataset), 20);
for i = randidx
    img = imread(dataset(i).img);
    jsdc = permute(dataset(i).jsdc, [2,3,1]);
    visualize_jsdc(img, jsdc);    
end

                                                                