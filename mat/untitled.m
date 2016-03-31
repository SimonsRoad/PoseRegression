close all;clc;clear;



% h = fspecial('gaussian', [32,16], 0.5);
% figure(2);imagesc(h); axis image;
% save('kernel.mat', 'h');


load('../src/testset_fcnlabel.mat');


data = permute(data,[3 4 2 1]);
label = permute(label,[3 4 2 1]);
resz = 1;

img = [];
for iSmp = 1:size(data,4) 
    img = data(:,:,:,iSmp);
    img = imresize(img, resz);
        
    clf;figure(1); set(gcf, 'Position', [600,800,600,1200]);
    subplot(4,4,1); imshow(imresize(img,0.25));
    for i = 1:14
        label_each = label(:,:,i,iSmp);
        label_each = imresize(label_each, resz);
        subplot(4,4,i+1); imagesc(label_each); axis image; colormap gray;
    end

end

