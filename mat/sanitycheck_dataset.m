% sanitycheck_dataset.m

clear; close all; clc;


pathtotxt = '~/develop/PoseRegression/data/rendout/tmp_y144_x256_aug/lists';
dataset = [];


% load image
fImage = fullfile(pathtotxt, 'img.txt');
fid = fopen(fImage);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    if cnt > 5100, break; end;
    dataset(cnt).img = tline;
    tline = fgetl(fid);
end
fclose(fid);

% load label
fLabel = fullfile(pathtotxt, 'jsdc.txt');
fid = fopen(fLabel);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    if cnt > 5100, break; end;
    tmp = load(tline);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    dataset(cnt).jsdc = tmp.jsdc;
    tline = fgetl(fid);                                                                                                                                                                                                                                                                                                                                                                                                                             
end
fclose(fid);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                
% visualize 
randidx = randi([1, numel(dataset)], 20);
for i = 1:numel(randidx)
    img = imread(dataset(i).img);
    jsdc = permute(dataset(i).jsdc, [2,3,1]);
    visualize_jsdc(img, jsdc);    
end

                                                                