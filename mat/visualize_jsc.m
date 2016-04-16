function visualize_jsdc(img, jsc, jsc_gt, nJoints)

part_str = {'htop', 'hbot', ...
    'lsho', 'lelb', 'lwr', 'lhip', 'lkne', 'lank', ...
    'rsho', 'relb', 'rwr', 'rhip', 'rkne', 'rank', ...
    'htop\_hbot', ...
    'hbot\_lsho', 'lsho\_lelb', 'lelb\_lwr', ...
    'lsho\_lhip', 'lhip\_lkne', 'lkne\_lank', ...
    'hbot\_rsho', 'rsho\_relb', 'relb\_rwr', ...
    'rsho\_rhip', 'rhip\_rkne', 'rkne\_rank', ...
    'seg', 'cen' ...
    };


%% hmap on the image
clf; figure(1);
for part = 1:27         % visualize j27
    
    % find peak (additional mark)
    tmp = jsc(:,:,part);
    [~,idx] = max(tmp(:));
    [y,x] = ind2sub(size(tmp), idx);
    
    % superimposing heat maps and image
    smap_aug = jsc(:,:,part);
    smap_aug(smap_aug<0) = 0;
    if max(smap_aug(:)) ~= 0 
        smap_aug = smap_aug / max(smap_aug(:));
    end    
    mapIm = mat2im(smap_aug, jet(100), [0 1]);
    
    imToShow{part} = mapIm*0.5 + (im2single(img))*0.5;
    subplot(4,8,part); hold on;
    imshow(imToShow{part}); 
    plot(x,y,'gx', 'LineWidth', 2); 
    title(part_str{part});
%     hold off;
    
    % mark ground-truth
    if part <nJoints+1
        tmp = jsc_gt(:,:,part);
        [~,idx] = max(tmp(:));
        [y,x] = ind2sub(size(tmp), idx);
        plot(x,y,'wx', 'LineWidth', 2); hold off;
    else
        hold off;
    end
end
for part = 28:29        % visualize sdc
    subplot(4,8,part); imagesc(jsc(:,:,part)); axis image;
    title(part_str{part});
end
for part = 30
    subplot(4,8,part); imshow(img);
    title('ori');
end


%% mesh plot
figure(2);
for part = 1:27         % visualize j27
    % superimposing heat maps and image
    smap_aug = jsc(:,:,part);
    subplot(4,8,part); mesh(smap_aug);
%     subplot(4,8,part); imagesc(smap_aug); axis image;
    title(part_str{part});
end
for part = 28:29        % visualize sc
    subplot(4,8,part); mesh(jsc(:,:,part));
%     subplot(4,8,part); imagesc(jsc(:,:,part)); axis image;
    title(part_str{part});
end


end