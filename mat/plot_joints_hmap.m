function plot_joints_hmap(img, hmap)

part_str = {'htop', 'hbot', ...
    'lsho', 'lelb', 'lwr', 'lhip', 'lkne', 'lank', ...
    'rsho', 'relb', 'rwr', 'rhip', 'rkne', 'rank', ...
    'htop\_hbot', ...
    'hbot\_lsho', 'lsho\_lelb', 'lelb\_lwr', ...
    'lsho\_lhip', 'lhip\_lkne', 'lkne\_lank', ...
    'hbot\_rsho', 'rsho\_relb', 'relb\_rwr', ...
    'rsho\_rhip', 'rhip\_rkne', 'rkne\_rank' ...
    };

figure(1);
for part = 1:size(hmap,3)

    % superimposing heat maps and image
    smap_aug = hmap(:,:,part);
    smap_aug(smap_aug<0) = 0;

    mapIm = mat2im(smap_aug, jet(100), [0 1]);

    imToShow{part} = mapIm*0.5 + (single(img)/255)*0.5;
    subplot(4,7,part); imshow(imToShow{part});
    title(part_str{part});
end

end