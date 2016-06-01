% vis_gt_img_jsc.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
% read some images and their jsc and save it as pdfs
% 
clc; close all; clear;



y = 235;
x = 325;

% img
imglist = sprintf('~/develop/PoseRegression/data/rendout/anc_y%03d_x%d/lists/img_sTest.txt', y,x);
fid = fopen(imglist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    dataset(cnt).im = tline;
    fname = sprintf('/home/namhoon/develop/PoseRegression/mat/gtexamples/pos/%03d.jpg', cnt);
    command = sprintf('cp %s %s', tline, fname);
%     system(command);
    tline = fgetl(fid);
end
fclose(fid);

% jsc
jsclist = sprintf('~/develop/PoseRegression/data/rendout/anc_y%03d_x%d/lists/jsc_sTest.txt', y,x);
fid = fopen(jsclist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    load(tline);
    jsc = permute(jsc, [2,3,1]);
    
    img = imread(dataset(cnt).im);
    
    for i = 1:size(jsc,3)
        smap_aug = jsc(:,:,i);
        smap_aug(smap_aug<0) = 0;
        mapIm = mat2im(smap_aug, jet(100), [0 1]);
        imToShow{i} = mapIm*0.5 + (single(img)/255)*0.5;
        figure(1); imshow(imToShow{i});
        print(gcf, sprintf('gtexamples/jsc/%03d_ch%02d',cnt,i), '-depsc');
    end
   
    
    % save into pdf
    
    tline = fgetl(fid);
end
fclose(fid);


