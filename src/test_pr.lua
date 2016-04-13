--[[
--test_pr.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--
-- set opt.nDonkeys 1 (no use of donkey, directly load data)
-- set opt.txtimg, opt.txtjsdc for real data. 
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
opt.nDonkeys = 1


-- load model
--local mNum  = 11
for mNum = 1, 1 do

    local mName = string.format('clear_model_%d.t7', mNum)
    --local pathToModel = '../save/PR_fcn/option/t_SunMar2721:48:402016'
    --local pathToModel = '../save/PR_fcn/option/t_TueMar2922:11:592016'
    local pathToModel = '../save/PR_fcn/option,LR=0.01/t_WedApr1320:28:332016'
    opt.retrain = paths.concat(pathToModel, mName)  

    local model, criterion = models.setup(opt)
    model:cuda()


    -- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
    --
    testsettype = 'rTest'

    savedir         = paths.concat(pathToModel, 'results/' .. testsettype)
    os.execute('mkdir -p ' .. savedir)
    savefile_img    = savedir..string.format('/img_model%d.mat', mNum)
    savefile_pred1   = savedir..string.format('/jsdc_pred_model%d_1.mat', mNum)
    savefile_pred2   = savedir..string.format('/jsdc_pred_model%d_2.mat', mNum)
    savefile_pred3   = savedir..string.format('/jsdc_pred_model%d_3.mat', mNum)
    savefile_pred4   = savedir..string.format('/jsdc_pred_model%d_4.mat', mNum)
    savefile_pred5   = savedir..string.format('/jsdc_pred_model%d_5.mat', mNum)
    savefile_gt     = savedir..string.format('/jsdc_gt_model%d.mat', mNum)

    if testsettype == 'sTrain' then
        testindices = torch.randperm(opt.nTrainData):index(1, torch.range(1,10):long())
        opt.txtimg  = '../data/rendout/tmp_y144_x256_aug/lists/img.txt'
        opt.txtjsdc = '../data/rendout/tmp_y144_x256_aug/lists/jsdc.txt'
    elseif testsettype == 'sTest' then
        testindices = torch.randperm(opt.nTestData):index(1, torch.range(1,10):long()) + opt.nTrainData
        opt.txtimg  = '../data/rendout/tmp_y144_x256_aug/lists/img.txt'
        opt.txtjsdc = '../data/rendout/tmp_y144_x256_aug/lists/jsdc.txt'
    elseif testsettype == 'rTest' then
        testindices = torch.range(1,22)
        opt.txtimg  = '../../towncenter/data/frames_y144_x256_sel/lists/img.txt'
        opt.txtjsdc = '../../towncenter/data/frames_y144_x256_sel/lists/jsdc.txt'
        opt.nJoints = 14
    end

    -- Load test data
    local loader = dataLoader{txtimg=opt.txtimg, txtjsdc=opt.txtjsdc}
    local testimg  = loader:load_img(testindices)
    local testjsdc = loader:load_jsdc(testindices)
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
    --matio.save(savefile_pred1, output[1]:float())
    --matio.save(savefile_pred2, output[2]:float())
    matio.save(savefile_pred1, output:float())
    matio.save(savefile_gt,   testjsdc) 


    -- compute PCK
    paths.dofile('eval_jsdc.lua')
    local gt_j27_hmap   = testjsdc[{ {}, {1,opt.nJoints}, {}, {} }]
    local pred_j27_hmap = output[{ {}, {1,opt.nJoints}, {}, {} }]
    local gt_j27   = find_peak(gt_j27_hmap)
    local pred_j27 = find_peak(pred_j27_hmap)
    local pck = comp_PCK(gt_j27, pred_j27)
    print(string.format('#model: %d | PCK: %.2f ', mNum, pck))
end
