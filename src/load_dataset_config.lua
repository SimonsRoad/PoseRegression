----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedestrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- load_dataset_config.lua
-- -
----------------------------------------------------------------------

function load_dataset_config(datasetname)
    if datasetname == 'towncenter' then
        YX = {{138,167}, {160,260}, {170,570}, {262,544}, {130,460}, {235,325}, {169,92}, {91,354}, {230,438}, {105,245}, {999,999}, {138,167}, {138,167}, {138,167}}
        WH = {{71,102},{76,109,},{78,112},{98,141},{69,99},{93,133},{78,112},{61,87},{91,131},{64,91},{78,112},{71,102},{71,102},{71,102}}
        --bestmodel = {1,5,3,6,6,4,3,9,5,8,11,33,22,30}
        datetime = {
            't_SunApr1705:37:282016', 
            't_TueApr1908:35:242016', 
            't_TueApr1922:59:002016', 
            't_WedApr2008:26:562016', 
            't_ThuApr2104:24:462016', 
            't_SunMay107:29:372016', 
            't_MonApr2504:23:352016', 
            't_MonApr2522:31:402016', 
            't_WedApr2709:26:522016',
            't_FriApr2917:18:122016',
            't_WedMay1119:15:512016',
            't_SatMay1409:44:292016',
            't_MonMay1600:50:302016',
            't_WedMay1807:57:252016'}
    elseif datasetname == 'pet2006' then
        YX = {{240,150}, {270,550}, {250,340}, {420,130}, {240,150}, {240,150}, {240,150}}
        WH = {{68,97}, {76,109}, {71,101}, {93,132}, {68,97}, {68,97}, {68,97}}
        --bestmodel = {20, 22, 28, 10, 27, 28}
        datetime = {
            't_ThuMay509:29:002016',
            't_FriMay609:55:552016',
            't_MonMay919:51:352016',
            't_SatMay702:56:182016',
            't_ThuMay1205:57:042016',
            't_FriMay1321:00:552016',
            't_SunMay1502:02:432016'}
    else
        error('invalid datasetname..!')
    end

end
