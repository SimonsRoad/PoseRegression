----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedestrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- test_fcn.lua
-- - This source code is originally created by Facebook, Inc.
----------------------------------------------------------------------

testLogger_epoch = optim.Logger(paths.concat(opt.save, 'test_epoch.log'))
testLogger_batch = optim.Logger(paths.concat(opt.save, 'test_batch.log'))
local loss_epoch


function test()
    print('--TEST STARTS..')

   	cutorch.synchronize()

   	-- set the dropouts to evaluate mode
   	model:evaluate()

    local tm = torch.Timer()
   	loss_epoch = 0

    -- randomize dataset (actually just indices)
    --local idx_rand = torch.randperm(opt.nTestData)
    local idx_rand = torch.range(1,opt.nTestData)

    -- test
   	for i=1,opt.nTestData/opt.batchSize do 
	  	local idx_start = (i-1) * opt.batchSize + 1
		local idx_end   = idx_start + opt.batchSize - 1
		local idx_batch
		if idx_end <= opt.nTestData then
            idx_batch = idx_rand[{{idx_start,idx_end}}]
		else
			local idx1 = idx_rand[{{idx_start,opt.nTestData}}]
			local idx2 = idx_rand[{{1,idx_end-opt.nTestData}}]
			idx_batch = torch.cat(idx1, idx2, 1)
		end
        idx_batch = idx_batch + opt.nTrainData      -- after # number of train.. 

        donkeys:addjob(
            function()
                return loader:load_batch_new(idx_batch)
            end,
            testBatch
        )
   	end

    donkeys:synchronize()
   	cutorch.synchronize()

   	loss_epoch = loss_epoch / (opt.nTestData/opt.batchSize) -- because loss is calculated per batch
   	testLogger_epoch:add{
		['avg loss (test set)'] = loss_epoch
   	}
   	print(string.format('Ep. [%d/%d]  (Test) Time(s): %.2f  ' .. 'avg loss (per batch): %.10f ', epoch, opt.nEpochs, tm:time().real, loss_epoch))

    collectgarbage()


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local outputs
local err

function testBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    collectgarbage()

   	inputs:resize(inputsCPU:size()):copy(inputsCPU)
   	labels:resize(labelsCPU:size()):copy(labelsCPU)

   	outputs = model:forward(inputs)
   	err = criterion:forward(outputs, labels)
   	cutorch.synchronize()

   	loss_epoch = loss_epoch + err

   	testLogger_batch:add{
		['avg loss (test batch)'] = err
   	}
    collectgarbage()

end
