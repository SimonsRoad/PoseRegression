function plot_pck_results_GenericGT()
% plot_pck_results_includingGT.m
%
clc; clear; close all;

% 1) towncenter_GenericGT 
% 2) pet2006_GT
datasetname = 'towncenter_GenericGT';     
datasettype = 'real';

pcks    = load_pck_results(datasetname, datasettype);
normscalors = 0:0.05:0.5;

pck_avg = pcks;


%% plot
if strcmp(datasetname, 'towncenter_GenericGT')
    pck_avg = pck_avg(:,[1,2,9,18,11,20]);
    opt = {
        {'-',   '>',  sprintf('SPN  [%.1f%%]', pck_avg(11,1))}, ...
        {'-',   '>',  sprintf('SPN-G [%.1f%%]', pck_avg(11,2))}, ...
        {'--',  'd',  sprintf('GT + CPM  [%.1f%%]', pck_avg(11,3))}, ...
        {'-.',  'd',  sprintf('GT + IEF  [%.1f%%]', pck_avg(11,4))}, ...
        {'--',  's',  sprintf('GT-J + CPM  [%.1f%%]', pck_avg(11,5))}, ...
        {'-.',  's',  sprintf('GT-J + IEF  [%.1f%%]', pck_avg(11,6))}, ...
        };
elseif strcmp(datasetname, 'pet2006_GT')
    pck_avg = pck_avg(:,[1,8,17,10,19]);
    opt = {
        {'-',   '>',  sprintf('SPN  [%.1f%%]', pck_avg(11,1))}, ...
        {'--',  'd',  sprintf('GT + CPM  [%.1f%%]', pck_avg(11,2))}, ...
        {'-.',  'd',  sprintf('GT + IEF  [%.1f%%]', pck_avg(11,3))}, ...
        {'--',  's',  sprintf('GT-J + CPM  [%.1f%%]', pck_avg(11,4))}, ...
        {'-.',  's',  sprintf('GT-J + IEF  [%.1f%%]', pck_avg(11,5))}, ...
        };
else
    error('invalid datasetname!');
end

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