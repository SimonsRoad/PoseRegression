function plot_detection_speed()

clc; close all; clear;


% speed = [0.17958 0.36586 0.56077 0.79052 0.94024 1.22192 1.36986];
speed = [1.36986 1.22192 0.94024 0.79052 0.56077 0.36586 0.17958 ];
method = {
    'DPM (VOC) + IEF', ...
    'DPM (VOC) + CPM', ...
    'DPM (INRIA) + IEF', ...
    'DPM (INRIA) + CPM', ...
    'R-CNN + IEF', ...
    'R-CNN + CPM', ...
    'ScenePoseNet'};

figure(1); 
barh(speed, 0.7);
for i=1:numel(speed)
    text(speed(i)+0.05,i, sprintf('%.2f',speed(i)),...
        'HorizontalAlignment', 'left', ...
        'FontSize', 11);
end
ax = gca;

% limit
xlim([0 max(speed)+0.2]) 
ylim([0.3 numel(speed)+0.5])
% x labels
set(ax,'YTickLabel',method);
% grid
ax.XGrid = 'on';
ax.XMinorGrid = 'on';
% tick
set(gca, 'Xtick',0:0.3:2);
% box
% ax.Box = 'on';
box off;
% labels
xlabel('speed (sec.)');
% font size
set(gca, 'FontSize', 14);


fname = 'otherplots/detectionspeed';
print(fname, '-depsc');


hold off;

end