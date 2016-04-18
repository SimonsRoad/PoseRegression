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

-- **SETTING**
opt.nDonkeys = 0


-- location
y   = 138
x   = 167


-- load model
for mNum = 1, 3 do

    local mName = string.format('clear_model_%d.t7', mNum)
    --local pathToModel = '../save/PR_fcn/option/t_SunMar2721:48:402016'
    --local pathToModel = '../save/PR_fcn/option/t_TueMar2922:11:592016'
    --local pathToModel = '../save/PR_fcn/option,LR=0.01/t_WedApr1320:28:332016'
    local pathToModel = '../save/PR_fcn/option/t_SunApr1709:33:532016'
    opt.retrain = paths.concat(pathToModel, mName)  

    local model, criterion = models.setup(opt)
    model:cuda()


    -- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
    --
    testsettype = 'rTest'

    savedir         = paths.concat(pathToModel, 'results/' .. testsettype)
    os.execute('mkdir -p ' .. savedir)
    savefile_img    = savedir..string.format('/img_model%d.mat', mNum)
    savefile_pred   = savedir..string.format('/jsc_pred_model%d.mat', mNum)
    savefile_gt     = savedir..string.format('/jsc_gt_model%d.mat', mNum)

    if testsettype == 'sTrain' then
        testindices = torch.randperm(opt.nTrainData):index(1, torch.range(1,20):long())
        opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../data/rendout/anc_y%d_x%d/lists/jsc_pos.txt', y, x)
    elseif testsettype == 'sTest' then
        testindices = torch.randperm(opt.nTestData):index(1, torch.range(1,20):long()) + opt.nTrainData
        opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../data/rendout/anc_y%d_x%d/lists/jsc_pos.txt', y, x)
    elseif testsettype == 'rTest' then
        testindices = torch.range(21,45)
        opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_sel/lists/img_pos.txt', y, x)
        opt.txtjsc  = string.format('../../towncenter/data/frames_y%d_x%d_sel/lists/jsc_pos.txt', y, x)
        opt.nJoints = 14
        -- this setting is actually cropping error. It needs to be gone.
        opt.W = 72
        opt.H = 103
        opt.W_jsc = 72
        opt.H_jsc = 103
    end

    -- Load test data
    local loader = dataLoader{txtimg=opt.txtimg, txtjsc=opt.txtjsc}
    local testimg = loader:load_img(testindices)
    local testjsc = loader:load_jsc(testindices)
    matio.save(savefile_img,  testimg)

    meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')
    meanstd = torch.load(meanstdCache)
    mean = meanstd.mean
    std  = meanstd.std
    for i=1,3 do
        testimg[{ {}, {i}, {}, {} }]:add(-mean[i])
        testimg[{ {}, {i}, {}, {} }]:div(std[i])
    end
    --print(testimg:size())


    -- Forward pass and save the results
    local output = model:forward(testimg:cuda())


    -- Save results
    matio.save(savefile_pred, output:float())
    matio.save(savefile_gt,   testjsc) 


    -- compute PCK
    paths.dofile('eval_jsc.lua')
    local gt_j27_hmap   = testjsc[{ {}, {1,opt.nJoints}, {}, {} }]
    local pred_j27_hmap = output[{ {}, {1,opt.nJoints}, {}, {} }]
    local gt_j27   = find_peak(gt_j27_hmap)
    local pred_j27 = find_peak(pred_j27_hmap)
    local pck = comp_PCK(gt_j27, pred_j27)
    print(string.format('#model: %d | PCK: %.2f ', mNum, pck))
end
