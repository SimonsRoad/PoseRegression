--[[
--load_dataset_config.lua
--Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]

function load_dataset_config(dataset)
    if dataset == 'towncenter' then
        YX = {{138,167}, {160,260}, {170,570}, {262,544}, {130,460}, {235,325}, {169,92}, {91,354}, {230,438}, {105,245}}
        WH = {{71,102},{76,109,},{78,112},{98,141},{69,99},{93,133},{78,112},{61,87},{91,131},{64,91}}
        --numtestimages = {45, 49, 29, 25, 23, 21, 75, 28, 16, 47}
        bestmodel = {1,5,3,6,6,4,3,9,5,8}
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
            't_FriApr2917:18:122016'}
    elseif dataset == 'pet2006' then
        YX = {{240,150}, {270,550}, {250,340}, {420,130}}
        WH = {{68,97}, {76,109}, {71,101}, {93,132}}
        --numimages = {288, 210, 297, 287}
        bestmodel = {20, 22, 28, 10}
        datetime = {
            't_ThuMay509:29:002016',
            't_FriMay609:55:552016',
            't_MonMay919:51:352016',
            't_SatMay702:56:182016'}
    else
        error('invalid dataset..!')
    end

end
