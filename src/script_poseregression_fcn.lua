--[[
--script_poseregression_fcn_new.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'

local models        = require 'models/init'
local opts          = require 'opts'
opt = opts.parse(arg)
print(opt)

torch.manualSeed(2)


-- Create model
model, criterion = models.setup(opt)


paths.dofile('data.lua')
paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')
paths.dofile('eval_jsdc.lua')

-- Training
epoch = opt.epochNumber
for i=1,opt.nEpochs do

	-- train and test
	train()
	test()

	-- evaluation
	if epoch % 1 == 0 then
        eval_jsdc()         -- evaluate on testset
	end
	epoch = epoch + 1
end



