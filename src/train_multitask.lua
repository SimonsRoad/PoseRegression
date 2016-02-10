--[[
-- train_multitask.lua
-- This code is originally from imigenet-multiGPU.torch by soumith
-- Modified by Namhoon Lee, (CMU, namhoonl@andrew.cmu.edu)
--]]

require 'optim'
require 'image'

local optimState = {
	learningRate = opt.LR,
	learningRateDecay = 0.0,
	momentum = opt.momentum,
	dampening = 0.0,
	weightDecay = opt.weightDecay
}


-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local loss_epoch


--3. Train - this function handles the high-level training loop,
--		     i.e. load data, train model, save model and state to disk
function train()

	batchNumber = 0
	cutorch.synchronize()

	-- set the dropouts to training mode
	--model:training()

	local tm = torch.Timer()
	loss_epoch = 0

	-- randomize dataset (actually just indices)
	local idx_rand = torch.randperm(trainset.label:size(1))

	for i=1, opt.epochSize do

		-- create indices for a batch
		local idx_start = batchNumber*opt.batchSize + 1
		local idx_end = idx_start + opt.batchSize - 1
		local idx_batch
		if idx_end <= trainset.label:size(1) then
			idx_batch = idx_rand[{{idx_start,idx_end}}] 
		else 
			local idx1 = idx_rand[{{idx_start,trainset.label:size(1)}}]
			local idx2 = idx_rand[{{1,idx_end-trainset.label:size(1)}}]
			idx_batch = torch.cat(idx1, idx2, 1)
		end
		
		-- loading "inputs" and "lables" for a batch
		local inputs, labels
		inputs = trainset.data:index(1, idx_batch:long())
		labels = trainset.label:index(1, idx_batch:long())

		-- check data-label matching (only test purpose)
		--local tmp = {data = inputs, label = labels}
		--save_tmp(tmp)
		--adf = adf + 1

		trainBatch(inputs, labels)
	end

	cutorch.synchronize()

	loss_epoch = loss_epoch / opt.epochSize

	trainLogger:add{
		['avg loss (train set)'] = loss_epoch
	}
	print(string.format('Ep. [%d/%d] ==> Total Time(s): %.2f  ' .. 'avg loss (per batch): %.6f ', epoch, opt.nEpochs, tm:time().real, loss_epoch))


	collectgarbage()

	local function sanitize(net)
		local list = net:listModules()
		for _,val in ipairs(list) do
			for name, field in pairs(val) do
				if torch.type(field) == 'cdata' then val[name] = nil end
				if (name == 'output' or name == 'gradInput') then
					val[name] = field.new()
				end
			end
		end
	end
	sanitize(model)
	saveDataParallel(paths.concat(opt.save, task .. 'model_' .. epoch .. '.t7'), model)
	torch.save(paths.concat(opt.save, task .. 'optimState_' .. epoch .. '.t7'), optimState)

end


local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()


-- 4. trainBatch -- used by train() to train a singel batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()

	-- transfer over to GPU
	inputs:resize(inputsCPU:size()):copy(inputsCPU)
	labels:resize(labelsCPU:size()):copy(labelsCPU)

	local err, outputs
	feval = function(x)
		model:zeroGradParameters()
		outputs = model:forward(inputs)

		---- multi task. (current: upper + lower concatanation)
		-- outputs1, labels1, err1, gradOutputs1 for upper body
		print(1)
		local idx_upper = torch.Tensor({1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22})
		print(2)
		local outputs1 = outputs:index(1, idx_upper:long())
		print(3)
		local labels1 = labels:index(1, idx_upper:long())
		print(labels1)
		local err1 = criterion1:forward(outputs1, labels1)
		print(err1)
		local gradOutputs1 = criterion1:backward(outputs1, labels1)
		print(gradOutputs1)
		
		-- outputs2, labels2, err2, gradOutputs2 for lower body
		local idx_lower = torch.Tensor({11,12,13,14,15,16,23,24,25,26,27,28})
		local outputs2 = outputs:index(1, idx_lower:long())
		local labels2 = labels:index(1, idx_lower:long())
		local err2 = criterion2:forward(outputs2, labels2)
		local gradOutputs2 = criterion2:backward(outputs2, labels2)

		-- gradOutputs = gradOutputs1 + gradOutputs2
		local gradOutputs = torch.cat(gradOutputs1, gradOutputs2)

		adf = adf + 1

		model:backward(inputs, gradOutputs)
		return err, gradParameters
	end
	optim.sgd(feval, parameters, optimState)

	-- DataParallelTabel's syncParameters
	model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

	cutorch.synchronize()
	batchNumber = batchNumber + 1
	loss_epoch = loss_epoch + err

	--print(('Ep. [%d/%d][%d/%d]\tTime %.3f Err %.5f LR %.0e DLTime %.3f'):format(epoch, opt.nEpochs, batchNumber, opt.epochSize, timer:time().real, err, optimState.learningRate, dataLoadingTime))

	dataTimer:reset()

end








