% cvt_channel_3to1.m

% load image
imgpath = '~/develop/PoseRegression/data/rendout/tmp_y144_x256_new/mask'; 
imgtype = '*.jpg';
images  = dir(fullfile(imgpath, imgtype));

% save image
imgpath_save = [imgpath, '_1ch'];
if ~exist(imgpath_save ,'dir'), mkdir(imgpath_save), end

for i = 1:numel(images)
    img = imread(fullfile(imgpath, images(i).name));
    img_gray = rgb2gray(img);
	img_double = im2double(img_gray);
    imwrite(img_double, fullfile(imgpath_save, images(i).name));
end
