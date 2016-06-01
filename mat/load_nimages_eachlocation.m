function nimages = load_nimages_eachlocation(datasetname, datasettype)

switch (datasetname)
    case 'towncenter'
        if strcmp(datasettype, 'real')
            nimages = [45 49 29 25 23 21 75 28 16 47];
        elseif strcmp(datasettype, 'synthetic')
            nimages = [100 100 100 100 100 100 100 100 100 100];
        else
            error('invalid dataset type! Should be either real or synthetic!');
        end
    case 'pet2006'
        if strcmp(datasettype, 'real')
            nimages = [288 210 297 284];
        elseif strcmp(datasettype, 'synthetic')
            nimages = [100 100 100 100];
        else
            error('invalid dataset type! Should be either real or synthetic!');
        end
    case 'towncenter_generic'
        if strcmp(datasettype, 'real')
            nimages = 351;
        elseif strcmp(datasettype, 'synthetic')
            nimages = 100;
        else
            error('invalid dataset type! Should be either real or synthetic!');
        end
    otherwise
        error('invalid datasetname!!!');
end

end