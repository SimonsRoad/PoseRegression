function plot_pck_results_ablative2()
% plot_pck_results_ablative2.m
% study for the effect of the number of SB moduels
%
clc; close all; clear;

datasetname = 'ablative2';     
datasettype = 'real';

pcks    = load_pck_results(datasetname, datasettype);
normscalors = 0:0.05:0.5;

pck_avg = pcks;


%% plot
opt = {
    {'-',  '>',  sprintf('SPN (3 SB)  [%.1f%%]', pck_avg(11,1))}, ...
    {'-',  'd',  sprintf('SPN (2 SB)  [%.1f%%]', pck_avg(11,2))}, ...
    {'-',  's',  sprintf('SPN (1 SB)  [%.1f%%]', pck_avg(11,3))}, ...
    };

CM = lines(numel(opt));     % color map
goplot(pck_avg, normscalors, CM, opt, datasetname, datasettype);


end

function goplot(pcks, normscalors, CM, opt, datasetname, datasettype)

% plot
figure(1); hold on;
for i=1:numel(opt)
    plot(normscalors, pcks(:,i), 'color', CM(i,:), 'LineWidth', 1.5, ...
        'LineStyle', opt{i}{1}, 'marker', opt{i}{2}, 'MarkerSize', 6);
end

% legend
for i=1:numel(opt)
    method{i} = opt{i}{3};
end
h_legend = legend(method, 'Location', 'best');
% grid
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
% tick
% ax.XTick = normscalors;
ax.XTick = 0:0.1:0.5;
% box
ax.Box = 'on';
% labels
xlabel('Normalized distance');
ylabel('Detection rate %');
% font size
set(ax, 'FontSize', 15);
set(h_legend, 'FontSize', 13);


% save
fname = sprintf('pckplots/%s_%s_wPCKh', datasetname,datasettype);
print(fname, '-depsc');

hold off;
end