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

local matio     = require 'matio'
local models    = require 'models/init'
local opts      = require 'opts'
opt = opts.parse(arg)


-- **SETTING**
opt.nDonkeys = 1


-- load model
local mNum  = 5
local mName = string.format('ori_model_%d.t7', mNum)
local pathToModel = '../save/PR_fcn/option/t_SunMar2709:01:552016'
opt.retrain = paths.concat(pathToModel, mName)  

model, criterion = models.setup(opt)
model:cuda()


-- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
--
testsettype = 'rTest'

savedir         = paths.concat(pathToModel, 'results/' .. testsettype)
savefile_img    = savedir..string.format('/img_model%d.mat', mNum)
savefile_pred   = savedir..string.format('/jsdc_pred_model%d.mat', mNum)
savefile_gt     = savedir..string.format('/jsdc_gt_model%d.mat', mNum)

if testsettype == 'sTrain' then
    testindices = torch.randperm(opt.nTrainData):index(1, torch.range(1,20):long())
    opt.txtimg  = '../data/rendout/tmp_y144_x256_aug/lists/img.txt'
    opt.txtjsdc = '../data/rendout/tmp_y144_x256_aug/lists/jsdc.txt'
elseif testsettype == 'sTest' then
    testindices = torch.randperm(opt.nTestData):index(1, torch.range(1,20):long()) + opt.nTrainData
    opt.txtimg  = '../data/rendout/tmp_y144_x256_aug/lists/img.txt'
    opt.txtjsdc = '../data/rendout/tmp_y144_x256_aug/lists/jsdc.txt'
elseif testsettype == 'rTest' then
    testindices = torch.range(1,22)
    opt.txtimg  = '../../towncenter/data/frames_y144_x256_sel/lists/img.txt'
    opt.txtjsdc = '../../towncenter/data/frames_y144_x256_sel/lists/jsdc.txt'
end

-- Load test data
paths.dofile('datanew.lua')
loader = dataLoader{txtimg=opt.txtimg, txtjsdc=opt.txtjsdc}
testimg = loader:load_img(testindices)
matio.save(savefile_img,  testimg)

meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')
meanstd = torch.load(meanstdCache)
mean = meanstd.mean
std  = meanstd.std
for i=1,3 do
    testimg[{ {}, {i}, {}, {} }]:add(-mean[i])
    testimg[{ {}, {i}, {}, {} }]:div(std[i])
end
print(testimg:size())


-- Forward pass and save the results
output = model:forward(testimg:cuda())


-- Save results
matio.save(savefile_pred, output:float())
matio.save(savefile_gt,   loader:load_jsdc(testindices)) 

-- compute PCK



