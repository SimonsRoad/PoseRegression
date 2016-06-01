% create_data_for_generic_detector_synthetic_towncenter.m
% Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
% Description:
% resize img and jsc and save
% 
clc; clear; close all;

% location for generic detector..
y = 999;
x = 999; 

testdata = [];
testdata.x = x;
testdata.y = y;

% target size (It's computed by averaging all different image sizes)
w_target = 78;
h_target = 112;


%% Note that the dataset is assumed to be synthetic! [sTest]
%% img
imglist = sprintf('~/develop/PoseRegression/data/rendout/anc_y%03d_x%d/lists_tmp/img_sTest.txt', y,x);
fid = fopen(imglist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    fprintf('cnt: %d\n',cnt);
    cnt = cnt + 1;
    testdata(cnt).im = tline;
    
    %% save as a new resized image
    % resize
    im_res = imresize(imread(testdata(cnt).im), [h_target w_target]);
    
    % save file name
    tmp = testdata(cnt).im;
    pos_a = strfind(tmp, 'anc_y');
    pos_pos = strfind(tmp, '/pos');
    pos_ped = strfind(tmp, '_Ped');
    savefilename = [tmp(1:pos_a-1), sprintf('anc_y%03d_x%d/pos_new',y,x), tmp(pos_ped-7:end)];
    imwrite(im_res, savefilename);
    tline = fgetl(fid);
end
fclose(fid);
%% jsc
jsclist = sprintf('~/develop/PoseRegression/data/rendout/anc_y%03d_x%d/lists_tmp/jsc_sTest.txt', y,x);
fid = fopen(jsclist);
tline = fgetl(fid);
cnt = 0;
while ischar(tline)
    fprintf('cnt: %d\n',cnt);
    cnt = cnt + 1;
    testdata(cnt).jsc = tline;
    
    %% save as a new resized jsc
    % resize
    load(testdata(cnt).jsc);
    jsc_perm = permute(jsc, [2,3,1]);
    jsc = single([]);
    for i=1:29
        tmp = imresize(jsc_perm(:,:,i), [h_target w_target]);
        jsc(:,:,i) = tmp;
    end
    jsc = permute(jsc,[3, 1, 2]);         % following Torch standard
    
    % sanity check
    %         plot_joints_hmap(imread(dataset_gt(cnt).im), jsc(:,:,1:27)); % sanity check
    % save file name
    tmp = testdata(cnt).jsc;
    pos_a = strfind(tmp, 'anc_y');
    pos_jsc = strfind(tmp, '/jsc');
    pos_ped = strfind(tmp, '_Ped');
    savefilename = [tmp(1:pos_a-1), sprintf('anc_y%03d_x%d/jsc_new',y,x), tmp(pos_ped-7:end)];
    save(savefilename, 'jsc');
    tline = fgetl(fid);
end
fclose(fid);
