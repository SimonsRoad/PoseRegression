clear; close all; clc;
testdata = [];


% img
imglist = '~/develop/PoseRegression/data/rendout/anc_y999_x999/lists/img_pos.txt';
fid = fopen(imglist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    testdata(cnt).im = tline;
    tline = fgetl(fid);
end
fclose(fid);

% jsc
jsclist = '~/develop/PoseRegression/data/rendout/anc_y999_x999/lists/jsc_pos.txt';
fid = fopen(jsclist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    cnt = cnt + 1;
    testdata(cnt).jsc = tline;
    load(testdata(cnt).jsc);
    jsc = permute(jsc, [2,3,1]);
    plot_joints_hmap(imread(testdata(cnt).im), jsc(:,:,1:27));
    tline = fgetl(fid);
end
fclose(fid);