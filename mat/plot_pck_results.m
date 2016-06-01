function plot_pck_results()
% plot_pck_results.m
%
close all; clear;

datasetname = 'pet2006';     % towncenter | pet2006 | towncenter_generic
datasettype = 'real';
includegt   = false;

pcks    = load_pck_results(datasetname, datasettype);
nimages = load_nimages_eachlocation(datasetname, datasettype);
nlocs   = numel(nimages);
normscalors = 0:0.05:0.5;

if nlocs > 1
    pck_avg = [];
    for i = 1:numel(normscalors)
        pcks_1scale = pcks((i-1)*nlocs+1:i*nlocs,:);
        pcks_1scale_avg = sum(pcks_1scale .* repmat(nimages', [1,size(pcks_1scale,2)]))/sum(nimages);
        pck_avg = [pck_avg; pcks_1scale_avg];
    end
else
    pck_avg = pcks;
end
assert(size(pck_avg,1)==numel(normscalors) & size(pck_avg,2)==19);


%% plot
if ~includegt
    pck_avg = pck_avg(:,[1,2,3,5,11,12,14]);
    opt = {
        {'-',   '>',  sprintf('ScenePoseNet [%.1f%%]', pck_avg(11,1))}, ...
        {'--',  '.',  sprintf('No detector + CPM [%.1f%%]', pck_avg(11,2))}, ...
        {'--',  'd',  sprintf('DPM + CPM [%.1f%%]', pck_avg(11,3))}, ...
        {'--',  's',  sprintf('R-CNN + CPM [%.1f%%]', pck_avg(11,4))}, ...
        {'-.',  '.',  sprintf('No detector + IEF [%.1f%%]', pck_avg(11,5))}, ...
        {'-.',  'd',  sprintf('DPM + IEF [%.1f%%]', pck_avg(11,6))}, ...
        {'-.',  's',  sprintf('R-CNN + IEF [%.1f%%]', pck_avg(11,7))}, ...
        };
else
    % include gt and gt jitter 0.1 
%     pck_avg = pck_avg(:,[1,2,3,4,5,8,10,11,12,13,14,17,19]);
%     opt = {
%         {'-',   '>',  sprintf('ScenePoseNet [%.1f%%]', pck_avg(11,1))}, ...
%         {'--',  '.',  sprintf('No detector + CPM [%.1f%%]', pck_avg(11,2))}, ...
%         {'--',  'd',  sprintf('DPM (INRIA) + CPM [%.1f%%]', pck_avg(11,3))}, ...
%         {'--',  'd',  sprintf('DPM (VOC) + CPM [%.1f%%]', pck_avg(11,4))}, ...
%         {'--',  's',  sprintf('R-CNN + CPM [%.1f%%]', pck_avg(11,5))}, ...
%         {'--',  'o',  sprintf('GT + CPM [%.1f%%]', pck_avg(11,6))}, ...
%         {'--',  'o',  sprintf('GT (jitter) + CPM [%.1f%%]', pck_avg(11,7))}, ...
%         {'-.',  '.',  sprintf('No detector + IEF [%.1f%%]', pck_avg(11,8))}, ...
%         {'-.',  'd',  sprintf('DPM (INRIA) + IEF [%.1f%%]', pck_avg(11,9))}, ...
%         {'-.',  'd',  sprintf('DPM (VOC) + IEF [%.1f%%]', pck_avg(11,10))}, ...
%         {'-.',  's',  sprintf('R-CNN + IEF [%.1f%%]', pck_avg(11,11))}, ...
%         {'-.',  'o',  sprintf('GT + IEF [%.1f%%]', pck_avg(11,12))}, ...
%         {'-.',  'o',  sprintf('GT (jitter) + IEF [%.1f%%]', pck_avg(11,13))}, ...
%         };
end
CM = lines(numel(opt));     % color map
goplot(pck_avg, normscalors, CM, opt, datasetname, datasettype, includegt);


end

function goplot(pcks, normscalors, CM, opt, datasetname, datasettype, includegt)

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
h_legend = legend(method, 'Location', 'northwest');
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
fname = sprintf('pckplots/%s_%s_gt%d_wPCKh', datasetname,datasettype,includegt);
print(fname, '-depsc');

hold off;
end