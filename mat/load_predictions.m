function dataset = load_predictions(dataset, datasettype, ablative)

nData = numel(dataset);
d = load_detector_info(dataset(1).y,dataset(1).x, ablative);

if strcmp(datasettype, 'rTest')
    for i=1:nData
        frm = dataset(i).im(end-7:end-4);
        f_jsc = fullfile(d.basedir, d.date, 'results', datasettype, sprintf('model%d/jsc_pred_frm%s.mat', d.bestmodel,frm));
        load(f_jsc);
        jsc = permute(x, [3,4,2,1]);
        dataset(i).jsc = jsc;
%         [dataset(i).point_pred, ~] = find_peak(jsc, 27);
        [dataset(i).point_pred, ~] = find_peak_advanced(jsc, 27, d.headsz);
        dataset(i).frm = str2num(frm);
    end
elseif strcmp(datasettype, 'sTest')
    for i=1:nData
        frm = i;
        f_jsc = fullfile(d.basedir, d.date, 'results', datasettype, sprintf('model%d/jsc_pred_frm%04d.mat', d.bestmodel,frm));
        load(f_jsc);
        jsc = permute(x, [3,4,2,1]);
        dataset(i).jsc = jsc;
        [dataset(i).point_pred, ~] = find_peak(jsc, 27);
%         [dataset(i).point_pred, ~] = find_peak_advanced(jsc, 27, d.headsz);
        dataset(i).frm = frm;
    end
else
    error('invalid datasettype!');
end


end