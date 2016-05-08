--[[
--test_pr.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--
-- set opt.nDonkeys 1 (no use of donkey, directly load data)
-- set opt.txtimg, opt.txtjsc for real data. 
-- set testindices properly..
-- set pathToModel
-- set mNum
-- test data needs to be pre-processed (resize!)
-- prepare a .txt for test data
--]]

require 'torch'
require 'paths'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'nngraph'

local matio     = require 'matio'
local models    = require 'models/init'
local opts      = require 'opts'
opt = opts.parse(arg)
paths.dofile('datanew.lua')
paths.dofile('eval_jsc.lua')

-- **SETTING**
opt.nDonkeys = 3


-- location
y   = 240
x   = 150
opt.W = 68
opt.H = 97
opt.W_jsc = opt.W
opt.H_jsc = opt.H


-- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
--
testsettype = 'rTest'
numimages   = 288
quality     = 'LQ'

local indices = torch.Tensor(1):long()

-- load model
for mNum = 1,12 do

    local mName = string.format('clear_model_%d.t7', mNum)

    --local pathToModel = '../save/PR_fcn/option/t_SunMar2721:48:402016'
    --local pathToModel = '../save/PR_fcn/option/t_TueMar2922:11:592016'
    --local pathToModel = '../save/PR_fcn/option,LR=0.01/t_WedApr1320:28:332016'
    --
    --local pathToModel = '../save/PR_fcn/option/t_SunApr1700:11:532016'
    --local pathToModel = '../save/PR_fcn/option/t_SunApr1705:37:282016'
    --local pathToModel = '../save/PR_fcn/option/t_SunApr1709:33:532016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr1810:11:492016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr1812:29:432016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr1820:55:372016'
    --local pathToModel = '../save/PR_fcn/option/t_TueApr1908:35:242016'
    --local pathToModel = '../save/PR_fcn/option/t_TueApr1922:59:002016'
    --local pathToModel = '../save/PR_fcn/option/t_WedApr2008:26:562016'
    --local pathToModel = '../save/PR_fcn/option/t_ThuApr2104:24:462016'
    --local pathToModel = '../save/PR_fcn/option/t_ThuApr2112:11:132016'
    --local pathToModel = '../save/PR_fcn/option/t_SunApr2407:18:482016'
    --local pathToModel = '../save/PR_fcn/option/t_SatApr2320:24:372016'
    --local pathToModel = '../save/PR_fcn/option/t_SunApr2419:20:302016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr2500:01:242016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr2504:23:352016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr2522:31:402016'
    --local pathToModel = '../save/PR_fcn/option/t_MonApr2522:54:292016'
    --local pathToModel = '../save/PR_fcn/option/t_TueApr2602:33:032016'
    --local pathToModel = '../save/PR_fcn/option/t_TueApr2608:26:562016'
    --local pathToModel = '../save/PR_fcn/option/t_TueApr2617:42:272016'
    --local pathToModel = '../save/PR_fcn/option/t_WedApr2709:26:522016'
    --local pathToModel = '../save/PR_fcn/option/t_ThuApr2807:24:202016'
    --local pathToModel = '../save/PR_fcn/option/t_FriApr2917:18:122016'
    --local pathToModel = '../save/PR_fcn/option/t_SunMay107:29:372016'
    --local pathToModel = '../save/PR_fcn/option/t_MonMay206:03:052016'
    --local pathToModel = '../save/PR_fcn/option/t_WedMay404:36:362016'
    --local pathToModel = '../save/PR_fcn/option/t_WedMay408:52:332016'
    --local pathToModel = '../save/PR_fcn/option/t_WedMay420:49:262016'
    --local pathToModel = '../save/PR_fcn/option/t_ThuMay509:29:002016'
    local pathToModel = '../save/PR_fcn/option/t_ThuMay522:48:142016'
    --local pathToModel = '../save/PR_fcn/option/t_FriMay609:55:552016'

    opt.retrain = paths.concat(pathToModel, mName)  
    local model, criterion = models.setup(opt)
    model:cuda()

    savedir = paths.concat(pathToModel, string.format('results/%s/model%d',testsettype,mNum))
    os.execute('mkdir -p ' .. savedir)

    if testsettype == 'sTrain' then
        testindices = torch.randperm(opt.nTrainData):index(1, torch.range(1,20):long())
        opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../data/rendout/anc_y%d_x%d/lists/jsc_pos.txt', y, x)
    elseif testsettype == 'sTest' then
        testindices = torch.randperm(opt.nTestData):index(1, torch.range(1,20):long()) + opt.nTrainData
        opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../data/rendout/anc_y%d_x%d/lists/jsc_pos.txt', y, x)
    elseif testsettype == 'rTest' then
        opt.txtimg  = string.format('../../pet2006/data/frames_y%d_x%d/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../../pet2006/data/frames_y%d_x%d/lists/jsc_pos.txt', y, x)
        --opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists/img_pos.txt', y, x, quality)
        --opt.txtjsc  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists/jsc_pos.txt', y, x, quality)
        --opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists_tmp/img_pos.txt', y, x, quality)
        --opt.txtjsc  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists_tmp/jsc_pos.txt', y, x, quality)
        opt.nJoints = 14
    end

    -- load meanstd
    meanstdCache = paths.concat(opt.cache, string.format('meanstdCache/y%d_x%d.t7',y,x))
    meanstd = torch.load(meanstdCache)
    mean = meanstd.mean
    std  = meanstd.std

    -- Test
    local loader = dataLoader{txtimg=opt.txtimg, txtjsc=opt.txtjsc}
    local pck_all = 0
    for i=1,numimages do
        -- load 
        indices[1] = i
        local testimg = loader:load_img(indices)
        local testjsc = loader:load_jsc(indices)
        for i=1,3 do
            testimg[{ {}, {i}, {}, {} }]:add(-mean[i])
            testimg[{ {}, {i}, {}, {} }]:div(std[i])
        end

        -- Forward pass and save the results
        local output = model:forward(testimg:cuda())

        -- compute PCK
        local gt_j27_hmap   = testjsc[{ {}, {1,opt.nJoints}, {}, {} }]
        local pred_j27_hmap = output[{ {}, {1,opt.nJoints}, {}, {} }]
        local gt_j27,occ   = find_peak(gt_j27_hmap)
        local pred_j27     = find_peak(pred_j27_hmap)
        local pck = comp_PCK(gt_j27, pred_j27, occ)
        pck_all = pck_all + pck
        --print(string.format('#model: %d | PCK (%dth image): %.2f ', mNum, i, pck))

        -- save output
        local savefile = savedir..string.format('/jsc_pred_frm%04d.mat', loader:fetch_framenumber(i))
        matio.save(savefile, output:float())
    end
    print(string.format('#model: %d | PCK (all images): %.2f (%%)' , mNum, pck_all/numimages))

end

-- Delete the directory created unnecessarily
sys.fexecute("rm -r "..opt.save)

