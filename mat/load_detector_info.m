function dinfo = load_detector_info(y,x,ablative)

dinfo.basedir = '~/develop/PoseRegression/save/PR_fcn/option';
if y == 138 && x == 167
    dinfo.w = 71; 
    dinfo.h = 102;
    switch(ablative)
        case 'no'
            dinfo.bestmodel = 1;
            dinfo.date = 't_SunApr1705:37:282016';
        case 'noprior'
            dinfo.bestmodel = 33;
            dinfo.date = 't_SatMay1409:44:292016';
        case 'more'
            dinfo.bestmodel = 22;
            dinfo.date = 't_MonMay1600:50:302016';
        case 'multiloss'
            dinfo.bestmodel = 30;
            dinfo.date = 't_WedMay1807:57:252016';
        otherwise
            error('invalid ablative option');
    end
    dinfo.headsz = 6.81;
elseif y == 160 && x == 260
    dinfo.w = 76;
    dinfo.h = 109;
    dinfo.bestmodel = 5;
    dinfo.date = 't_TueApr1908:35:242016';
    dinfo.headsz = 7.63;
elseif y == 170 && x == 570
    dinfo.w = 78;
    dinfo.h = 112;
    dinfo.bestmodel = 3;
    dinfo.date = 't_TueApr1922:59:002016';
    dinfo.headsz = 7.72;
elseif y == 262 && x == 544
    dinfo.w = 98;
    dinfo.h = 141;
    dinfo.bestmodel = 6;
    dinfo.date = 't_WedApr2008:26:562016';
    dinfo.headsz = 9.54;
elseif y == 130 && x == 460
    dinfo.w = 69;
    dinfo.h = 99;
    dinfo.bestmodel = 6;
    dinfo.date = 't_ThuApr2104:24:462016';
    dinfo.headsz = 6.94;
elseif y == 235 && x == 325
    dinfo.w = 93;
    dinfo.h = 133;
    dinfo.bestmodel = 4;
    dinfo.date = 't_SunMay107:29:372016';
    dinfo.headsz = 9.12;
elseif y == 169 && x == 92
    dinfo.w = 78;
    dinfo.h = 112;
    dinfo.bestmodel = 3;
    dinfo.date = 't_MonApr2504:23:352016';
    dinfo.headsz = 7.92;
elseif y == 91  && x == 354
    dinfo.w = 61;
    dinfo.h = 87;
    dinfo.bestmodel = 9;
    dinfo.date = 't_MonApr2522:31:402016';
    dinfo.headsz = 5.83;
elseif y == 230 && x == 438
    dinfo.w = 91;
    dinfo.h = 131;
    dinfo.bestmodel = 5;
    dinfo.date = 't_WedApr2709:26:522016';
    dinfo.headsz = 9.17;
elseif y == 105 && x == 245
    dinfo.w = 64;
    dinfo.h = 91;
    dinfo.bestmodel = 8;
    dinfo.date = 't_FriApr2917:18:122016';
    dinfo.headsz = 6.35;
elseif y == 240 && x == 150
    dinfo.w = 68;
    dinfo.h = 97;
    switch(ablative)
        case 'no'
            dinfo.bestmodel = 20;
            dinfo.date = 't_ThuMay509:29:002016';
        case 'CF1'
            dinfo.bestmodel = 27;
            dinfo.date = 't_ThuMay1205:57:042016';
        case 'CF2'
            dinfo.bestmodel = 28;
            dinfo.date = 't_FriMay1321:00:552016';
        case 'nosegcen'
            dinfo.bestmodel = 33;
            dinfo.date = 't_SunMay1502:02:432016';
        otherwise
            error('invalid ablative option');
    end
    dinfo.headsz = 6.73;
elseif y == 270 && x == 550
    dinfo.w = 76;
    dinfo.h = 109;
    dinfo.bestmodel = 22;
    dinfo.date = 't_FriMay609:55:552016';
    dinfo.headsz = 7.40;
elseif y == 250 && x == 340
    dinfo.w = 71;
    dinfo.h = 101;
    dinfo.bestmodel = 28;
    dinfo.date = 't_MonMay919:51:352016';
    dinfo.headsz = 7.09;
elseif y == 420 && x == 130
    dinfo.w = 93;
    dinfo.h = 132;
    dinfo.bestmodel = 10;
    dinfo.date = 't_SatMay702:56:182016';
    dinfo.headsz = 9.21;
elseif y == 999 && x == 999
    dinfo.w = 78;
    dinfo.h = 112;
    dinfo.bestmodel = 11;
    dinfo.date = 't_WedMay1119:15:512016';
    dinfo.headsz = 7.72;
else
    error('No available anchor location!');
end


end
