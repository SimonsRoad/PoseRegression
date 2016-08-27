----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedestrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- runforheatmaps.lua
-- -
----------------------------------------------------------------------

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
y   = 105
x   = 245
opt.W = 64
opt.H = 91
opt.W_jsc = opt.W
opt.H_jsc = opt.H


-- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
--
testsettype = 'rTest'
quality     = 'LQ'

local indices = torch.Tensor(1):long()

-- load model
mNum = 4
local mName = string.format('clear_model_%d.t7', mNum)

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
--local pathToModel = '../save/PR_fcn/option/t_TueApr2602:33:032016'
--local pathToModel = '../save/PR_fcn/option/t_TueApr2608:26:562016'
--local pathToModel = '../save/PR_fcn/option/t_TueApr2617:42:272016'
--local pathToModel = '../save/PR_fcn/option/t_WedApr2709:26:522016'
--local pathToModel = '../save/PR_fcn/option/t_SunMay107:29:372016'
local pathToModel = '../save/PR_fcn/option/t_FriApr2917:18:122016'

opt.retrain = paths.concat(pathToModel, mName)  
local model, criterion = models.setup(opt)
model:cuda()

savedir = paths.concat(pathToModel, string.format('results_multi/%s/model%d',testsettype,mNum))
os.execute('mkdir -p ' .. savedir)

if testsettype == 'sTrain' then
    testindices = torch.randperm(opt.nTrainData):index(1, torch.range(1,20):long())
    opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
elseif testsettype == 'sTest' then
    testindices = torch.randperm(opt.nTestData):index(1, torch.range(1,20):long()) + opt.nTrainData
    opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
elseif testsettype == 'rTest' then
    --opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists/img_pos_multi.txt', y, x, quality)
    opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_%s/lists/img_pos.txt', y, x, quality)
end

-- load meanstd
local meanstdCache = paths.concat(opt.cache, string.format('meanstdCache/y%d_x%d.t7',y,x))
local meanstd = torch.load(meanstdCache)
local mean = meanstd.mean
local std  = meanstd.std

-- Test (warm up the model before measuring time!!)
for i=1,3 do
    print('warming up..')
    local testinput  = torch.rand(1,3,opt.H, opt.W):cuda()
    local testoutput = model:forward(testinput)
end
--local loader = dataLoader{txtimg=opt.txtimg, txtjsc=opt.txtjsc}
local loader = dataLoader{txtimg=opt.txtimg}
local timer = torch.Timer()
local t_total = 0
for i=1,loader:size() do
    print(string.format('processing image %d..', i))

    timer:reset()
    -- load image
    indices[1] = i
    local testimg = loader:load_img(indices)
    for i=1,3 do
        testimg[{ {}, {i}, {}, {} }]:add(-mean[i])
        testimg[{ {}, {i}, {}, {} }]:div(std[i])
    end

    -- Forward pass and save the results
    local output = model:forward(testimg:cuda())

    -- get peak. Just need to compute the total time for our algorithm
    local pred_j27_hmap = output[{ {}, {1,27}, {}, {} }]
    local pred_j27 = find_peak(pred_j27_hmap)
    local t = timer:time().real
    t_total = t_total + t

    -- save output 
    --local savefile = savedir..string.format('/jsc_pred_frm%04d.mat', loader:fetch_framenumber(i))
    --matio.save(savefile, output:float())
end
print(string.format('total processing time: %.4f ', t_total))
print(string.format('  avg processing time: %.4f ', t_total/loader:size()))
print(loader:size())


-- Delete the directory created unnecessarily
sys.fexecute("rm -r "..opt.save)
