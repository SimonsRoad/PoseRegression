function outpose = convert_pose_original(pose, cropinfo)

% cropinfo has,
% sz_new_img, sz_orig_img
% min_x, min_y, max_x, max_y, padd_x_1, padd_y_1

sz_new_img = cropinfo.sz_new_img;
sz_orig_img = cropinfo.sz_orig_img;
sz_padd_img = cropinfo.sz_padd_img;
img_side = cropinfo.img_side;
min_x = cropinfo.min_x;
min_y = cropinfo.min_y;
max_x = cropinfo.max_x;
max_y = cropinfo.max_y;
padd_x_1 = cropinfo.padd_x_1;
padd_y_1 = cropinfo.padd_y_1;


% (reverse) resize: img_side -> new_img
outpose = pose*(sz_new_img(1)/img_side);

% (reverse) truncate: new_img -> padd_img
outpose = outpose + repmat([min_x+padd_x_1 min_y+padd_y_1], [size(outpose,1),1]);

% (reverse) padding: padd_img -> orig_img
outpose = outpose - repmat([padd_x_1 padd_y_1], [size(outpose,1),1]);

end