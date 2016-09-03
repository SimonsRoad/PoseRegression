----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- script_poseregression_fcn.lua
-- - main 
----------------------------------------------------------------------

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
print('-------------------------------------------------------------------')
print('TEST!!!')
print(model)
print(model:get(2))
print(model:get(2):listModules())
print(model:get(2).output:size())
for i=1,100 do
    print(i, model:get(i))
end

-- test forward pass
testinput = torch.rand(1,3,opt.H,opt.W):cuda()
output = model:forward(testinput)
print(model:get(12):listModules())
--print(model:get(53):listModules())
--print(model:get(68):listModules())
--print(model:get(83):listModules())
--print(model:get(90):listModules())
--print(model:get(96):listModules())
--print(model:get(53).output:size())

--print(model:get(2).output:size())
--print(model:get(2).weight:size())
--print(model.innode)
--print(model.outnode)
--print(model.nInputs)
--print(model._type)
--print(model.outnode)
--print(model.outnode.data)
--print(model.innode.data.mapindex)
--print(model.outnode.data.mapindex[1])
--print(model.outnode.data.mapindex[1].module)
print('-------------------------------------------------------------------')
adf=adf+1

paths.dofile('data.lua')
paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')
paths.dofile('eval_jsc.lua')

-- Training
epoch = opt.epochNumber
for i=1,opt.nEpochs do
	train()
	test()
    eval_jsc()         -- evaluate on testset
	epoch = epoch + 1
end



