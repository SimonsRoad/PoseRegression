function visualize_jsdc(img, jsdc, jsdc_gt)

part_str = {'htop', 'hbot', ...
    'lsho', 'lelb', 'lwr', 'lhip', 'lkne', 'lank', ...
    'rsho', 'relb', 'rwr', 'rhip', 'rkne', 'rank', ...
    'htop\_hbot', ...
    'hbot\_lsho', 'lsho\_lelb', 'lelb\_lwr', ...
    'lsho\_lhip', 'lhip\_lkne', 'lkne\_lank', ...
    'hbot\_rsho', 'rsho\_relb', 'relb\_rwr', ...
    'rsho\_rhip', 'rhip\_rkne', 'rkne\_rank', ...
    'seg', 'dep', 'cen' ...
    };

clf; figure(1);
for part = 1:27         % visualize j27
    
    % find peak (additional mark)
    tmp = jsdc(:,:,part);
    [~,idx] = max(tmp(:));
    [y,x] = ind2sub(size(tmp), idx);
    
    % superimposing heat maps and image
    smap_aug = jsdc(:,:,part);
    smap_aug(smap_aug<0) = 0;
%     smap_aug = smap_aug / max(smap_aug(:));
    
    mapIm = mat2im(smap_aug, jet(100), [0 1]);
    
    imToShow{part} = mapIm*0.5 + (single(img))*0.5;
    subplot(4,8,part); hold on;
    imshow(imToShow{part}); 
    plot(x,y,'r*'); 
    title(part_str{part});
    
    % mark ground-truth
%     if part <15
%         tmp = jsdc_gt(:,:,part);
%         [~,idx] = max(tmp(:));
%         [y,x] = ind2sub(size(tmp), idx);
%         plot(x,y,'g*'); hold off;
%     else
%         hold off;
%     end
end
for part = 28:30        % visualize sdc
    subplot(4,8,part); imshow(jsdc(:,:,part));
    title(part_str{part});
end
for part = 31
    subplot(4,8,part); imshow(img);
    title('ori');
end

figure(2);
for part = 1:27         % visualize j27
    % superimposing heat maps and image
    smap_aug = jsdc(:,:,part);
    subplot(4,8,part); mesh(smap_aug);
    title(part_str{part});
end
for part = 28:30        % visualize sdc
    subplot(4,8,part); mesh(jsdc(:,:,part));
    title(part_str{part});
end


end