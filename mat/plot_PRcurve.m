function plot_PRcurve()
close all; clear;
datasetname = 'towncenter';
datasettype = 'rTest';
detectionmethod = {'OURS', 'DPM_INRIA', 'RCNN'};

if strcmp(datasetname, 'towncenter')
    if strcmp(datasettype, 'rTest')
        total_gt_rects = 316;
    elseif strcmp(datasettype, 'sTest')
        total_gt_rects = 1000;
    end
elseif strcmp(datasetname, 'pet2006')
    if strcmp(datasettype, 'rTest')
        total_gt_rects = 1079;
    elseif strcmp(datasettype, 'sTest')
        total_gt_rects = 400;
    end
end


%% PR
for i=1:numel(detectionmethod)
    fname = sprintf('detectionscores/%s_%s/PR_%s.mat', datasetname,datasettype,detectionmethod{i});
    load(fname);
    p{i} = precision;
    r{i} = recall;
    
    % ap
    sel = find(diff(recall)) + 1;
    ap{i} = sum(precision(sel)) / total_gt_rects;
end




%% Plot 
opt = {
    {'-',   '>',  sprintf('SPN [AP: %.3f]', ap{1})}, ...
    {'--',  '>',  sprintf('DPM [AP: %.3f]', ap{2})}, ...
    {'-.',  'd',  sprintf('RCNN [AP: %.3f]', ap{3})}, ...
    };
CM = lines(numel(opt));     % color map


figure(1); hold on; grid on; 
axis square
for i=1:numel(detectionmethod)
    plot(r{i}, p{i}, 'color', CM(i,:), 'LineWidth', 3, 'LineStyle', opt{i}{1}); 
end
% legend
for i=1:numel(opt)
    method{i} = opt{i}{3};
end
h_legend = legend(method, 'Location', 'southeast');
% grid
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
% limit
ylim([0 1]);
% tick
% ax.XTick = normscalors;
ax.XTick = 0:0.2:1.0;
% box
ax.Box = 'on';
% labels
xlabel('Recall'); ylabel('Precision');
% font size
set(ax, 'FontSize', 20);
set(h_legend, 'FontSize', 17);

% save
fname = sprintf('PRcurve/%s_%s', datasetname,datasettype);
print(fname, '-depsc', '-r0');

hold off;


end








