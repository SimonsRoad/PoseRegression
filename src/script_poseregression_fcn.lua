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
local checkpoints   = require 'checkpoints'

opt = opts.parse(arg)

paths.dofile('util.lua')
paths.dofile('compute_distance.lua')
paths.dofile('evaluate.lua')
paths.dofile('load_batch.lua')
paths.dofile('eval_jsdc.lua')


-- 0. settings 
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')
print('Saving everything to: ' .. opt.save) 
os.execute('mkdir -p ' .. opt.save)


-- Create model
model, criterion = models.setup(opt, checkpoint)


-- 4. Training
--
print(opt)

paths.dofile('data.lua')
paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')

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




