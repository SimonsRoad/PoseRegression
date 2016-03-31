function plot_joints(img, joints1, joints2)
fprintf('img size: [%d, %d, %d] \n', size(img));

figure; imshow(img); hold on;
if nargin < 3
    for i=1:size(joints1,1)
        plot(joints1(i,1), joints1(i,2), 'r*');
        drawnow;
        pause(0.2);
    end
else
    for i=1:size(joints1,1)
        plot(joints1(i,1), joints1(i,2), 'r*');     % gt
        plot(joints2(i,1), joints2(i,2), 'g*');     % est
        drawnow;
        pause(0.2);
    end
end
end