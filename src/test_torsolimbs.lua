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

	local nTest = testset.label:size(1)

   	loss_epoch = 0
   	for i=1,nTest/opt.batchSize do 
	  	 local idx_start = i * opt.batchSize + 1
		 local idx_end   = idx_start + opt.batchSize - 1
		 local idx_batch
		 if idx_end <= nTest then
			 idx_batch = torch.range(idx_start, idx_end)
		 else
			 local idx1 = torch.range(idx_start, nTest)
			 local idx2 = torch.range(1, idx_end-nTest)
			 idx_batch = torch.cat(idx1, idx2, 1)
		 end

		 local inputs, labels
		 inputs = testset.data:index(1, idx_batch:long())
		 labels = testset.label:index(1, idx_batch:long())

         testBatch(inputs, labels)
   	end

   	cutorch.synchronize()

   	loss_epoch = loss_epoch / (nTest/opt.batchSize) -- because loss is calculated per batch
   	testLogger:add{
		['avg loss (test set)'] = loss_epoch
   	}
   	print(string.format('Ep. [%d/%d]  (Test) Time(s): %.2f  ' .. 'avg loss (per batch): %.8f ', epoch, opt.nEpochs, timer:time().real, loss_epoch))


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local inputs1 = torch.CudaTensor()
local inputs2 = torch.CudaTensor()
local labels = torch.CudaTensor()
local labels1 = torch.CudaTensor()
local labels2 = torch.CudaTensor()


function testBatch(inputsCPU, labelsCPU)
   	inputs:resize(inputsCPU:size()):copy(inputsCPU)
   	labels:resize(labelsCPU:size()):copy(labelsCPU)

   	local outputs = model:forward(inputs)
	local outputs1 = outputs[1]
	local outputs2 = outputs[2]

	local idx_torso = torch.Tensor({1,2,3,4,5,6,11,12,17,18,23,24})
	local idx_limbs = torch.Tensor({7,8,9,10,13,14,15,16,19,20,21,22,25,26,27,28})
	local labels1 = labels:index(2, idx_torso:long())
	local labels2 = labels:index(2, idx_limbs:long())

   	local err = criterion:forward({outputs1, outputs2}, {labels1, labels2})
   	cutorch.synchronize()

   	loss_epoch = loss_epoch + err

end
