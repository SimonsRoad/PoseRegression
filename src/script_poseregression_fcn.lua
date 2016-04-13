--[[
--script_poseregression_fcn.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--namhoonl@andrew.cmu.edu
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'

local models        = require 'models/init'
local opts          = require 'opts'
opt = opts.parse(arg)
print(opt)

--torch.manualSeed(2)


-- Create model
model, criterion = models.setup(opt)

paths.dofile('data.lua')
paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')
paths.dofile('eval_jsdc.lua')

-- Training
epoch = opt.epochNumber
for i=1,opt.nEpochs do
	train()
	test()
    eval_jsdc()         -- evaluate on testset
	epoch = epoch + 1
end



