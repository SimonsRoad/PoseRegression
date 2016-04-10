clf; close all; clc; clear;
load('testjsdc.mat');
hmap = jsdc(:,:,1:27);
background = 1-jsdc(:,:,28);
figure(1); hold on;
for i=1:27
    subplot(4,7,i); imagesc(hmap(:,:,i)); axis image;
end
subplot(4,7,28); imagesc(background); axis image;

%% normalize
catchannel = cat(3, hmap, background);
sumchannel = repmat(sum(catchannel,3), [1,1,28]);
out = catchannel ./ sumchannel; 
figure(2); hold on;
for i=1:28
    subplot(4,7,i); imagesc(out(:,:,i)); axis image;
end
