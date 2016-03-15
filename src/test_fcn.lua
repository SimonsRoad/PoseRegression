--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  **Modified by Namhoon Lee (namhoonl@andrew.cmu.edu), RI CMU
--
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local loss_epoch
local timer = torch.Timer()

function test()

   	cutorch.synchronize()
   	timer:reset()

   	-- set the dropouts to evaluate mode
   	model:evaluate()

   	loss_epoch = 0
   	for i=1,opt.nTestData/opt.batchSize do 
	  	local idx_start = (i-1) * opt.batchSize + 1
		local idx_end   = idx_start + opt.batchSize - 1
		local idx_batch
		if idx_end <= opt.nTestData then
			idx_batch = torch.range(idx_start, idx_end)
		else
			local idx1 = torch.range(idx_start, opt.nTestData)
			local idx2 = torch.range(1, idx_end-opt.nTestData)
			idx_batch = torch.cat(idx1, idx2, 1)
		end

        donkeys:addjob(
            function()
                local testset_batch = loader:load_batch(idx_batch)
                return testset_batch.data, testset_batch.label
            end,
            testBatch
        )
   	end

    donkeys:synchronize()
   	cutorch.synchronize()

   	loss_epoch = loss_epoch / (opt.nTestData/opt.batchSize) -- because loss is calculated per batch
   	testLogger:add{
		['avg loss (test set)'] = loss_epoch
   	}
   	print(string.format('Ep. [%d/%d]  (Test) Time(s): %.2f  ' .. 'avg loss (per batch): %.8f ', epoch, opt.nEpochs, timer:time().real, loss_epoch))


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()


function testBatch(inputsCPU, labelsCPU)

   	inputs:resize(inputsCPU:size()):copy(inputsCPU)
   	labels:resize(labelsCPU:size()):copy(labelsCPU)

   	local outputs = model:forward(inputs)
   	local err = criterion:forward(outputs, labels)
   	cutorch.synchronize()

   	loss_epoch = loss_epoch + err

end
