--[[
--test_pr.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--
-- set opt.nDonkeys 1 (no use of donkey, directly load data)
-- set testindices properly..
-- set opt.retrain with a saved model to test
-- set opt.txtpos
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
print(opt)


-- **SETTING**
opt.nDonkeys = 1
-- synthetic
opt.txttest = '../data/rendout/tmp_y144_x256_aug/lists/pos.txt'
testindices = torch.range(1, 20)
savefile = '../save/testdir/jsdc_s20_train.mat'
-- real
--opt.txttest = '../data/lists/frames_y144_x256_sel.txt'
--testindices = torch.range(1,22)
--savefile = '../save/testdir/jsdc_r22.mat'


-- Create model
model, criterion = models.setup(opt)


-- Load test data
paths.dofile('loadtestset.lua')
loader = dataLoader{txttest=opt.txttest}

testimg = loader:load_img(testindices)
print(testimg:size())

-- Load mean&std
meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')
meanstd = torch.load(meanstdCache)
mean = meanstd.mean
std  = meanstd.std
for i=1,3 do
    testimg[{ {}, {i}, {}, {} }]:add(-mean[i])
    testimg[{ {}, {i}, {}, {} }]:div(std[i])
end


-- Forward pass and save the results
output = model:forward(testimg:cuda())
matio.save(savefile, output:float())
