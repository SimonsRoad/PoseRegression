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

if opt.optimState ~= 'none' then
	assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
	print('Loading optimState from file: ' .. opt.optimState)
	optimState = torch.load(opt.optimState)
end


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
	model:training()

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

		trainBatch(inputs, labels)
	end

	cutorch.synchronize()

	loss_epoch = loss_epoch / opt.epochSize

	trainLogger:add{
		['avg loss (train set)'] = loss_epoch
	}
	print(string.format('Ep. [%d/%d] ==> Total Time(s): %.2f  ' .. 'avg loss (per batch): %.8f ', epoch, opt.nEpochs, tm:time().real, loss_epoch))


	collectgarbage()

	local function sanitize(net)
		local list = net:listModules()
		for _,val in ipairs(list) do
			for name, field in pairs(val) do
				if torch.type(field) == 'cdata' then val[name] = nil end
				--if (name == 'output' or name == 'gradInput') then
				--	val[name] = field.new()
				--end
			end
		end
	end
	--sanitize(model)
	--model:clearState()
	if epoch % 500 == 0 then
		--model:clearState()
		saveDataParallel(paths.concat(opt.save, opt.t .. '_model_' .. epoch .. '.t7'), model)
		torch.save(paths.concat(opt.save, opt.t .. '_optimState_' .. epoch .. '.t7'), optimState)
	end

end


local inputs = torch.CudaTensor()
local inputs1 = torch.CudaTensor()
local inputs2 = torch.CudaTensor()
local inputs3 = torch.CudaTensor()
local inputs4 = torch.CudaTensor()
local inputs5 = torch.CudaTensor()
local inputs6 = torch.CudaTensor()
local inputs7 = torch.CudaTensor()
local inputs8 = torch.CudaTensor()
local inputs9 = torch.CudaTensor()
local inputs10 = torch.CudaTensor()
local inputs11 = torch.CudaTensor()
local inputs12 = torch.CudaTensor()
local inputs13 = torch.CudaTensor()
local inputs14 = torch.CudaTensor()

local labels = torch.CudaTensor()
local labels1 = torch.CudaTensor()
local labels2 = torch.CudaTensor()
local labels3 = torch.CudaTensor()
local labels4 = torch.CudaTensor()
local labels5 = torch.CudaTensor()
local labels6 = torch.CudaTensor()
local labels7 = torch.CudaTensor()
local labels8 = torch.CudaTensor()
local labels9 = torch.CudaTensor()
local labels10 = torch.CudaTensor()
local labels11 = torch.CudaTensor()
local labels12 = torch.CudaTensor()
local labels13 = torch.CudaTensor()
local labels14 = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()


-- 4. trainBatch -- used by train() to train a singel batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()

	-- idx for joints
	local htop = torch.Tensor({1,2})
	local hbot = torch.Tensor({3,4})
	local lsho = torch.Tensor({5,6})
	local lelb = torch.Tensor({7,8})
	local lwr  = torch.Tensor({9,10})
	local lhip = torch.Tensor({11,12})
	local lkne = torch.Tensor({13,14})
	local lank = torch.Tensor({15,16})
	local rsho = torch.Tensor({17,18})
	local relb = torch.Tensor({19,20})
	local rwr  = torch.Tensor({21,22})
	local rhip = torch.Tensor({23,24})
	local rkne = torch.Tensor({25,26})
	local rank = torch.Tensor({27,28})

	-- transfer over to GPU
	inputs:resize(inputsCPU:size()):copy(inputsCPU)
	labels:resize(labelsCPU:size()):copy(labelsCPU)

	local err, outputs
	feval = function(x)
		model:zeroGradParameters()
		outputs = model:forward(inputs)

		local outputs1 = outputs[1]
		local outputs2 = outputs[2]
		local outputs3 = outputs[3]
		local outputs4 = outputs[4]
		local outputs5 = outputs[5]
		local outputs6 = outputs[6]
		local outputs7 = outputs[7]
		local outputs8 = outputs[8]
		local outputs9 = outputs[9]
		local outputs10 = outputs[10]
		local outputs11 = outputs[11]
		local outputs12 = outputs[12]
		local outputs13 = outputs[13]
		local outputs14 = outputs[14]
		local labels1 = labels:index(2, htop:long())
		local labels2 = labels:index(2, hbot:long())
		local labels3 = labels:index(2, lsho:long())
		local labels4 = labels:index(2, lelb:long())
		local labels5 = labels:index(2, lwr:long())
		local labels6 = labels:index(2, lhip:long())
		local labels7 = labels:index(2, lkne:long())
		local labels8 = labels:index(2, lank:long())
		local labels9 = labels:index(2, rsho:long())
		local labels10 = labels:index(2, relb:long())
		local labels11 = labels:index(2, rwr:long())
		local labels12 = labels:index(2, rhip:long())
		local labels13 = labels:index(2, rkne:long())
		local labels14 = labels:index(2, rank:long())
		err = criterion:forward({outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, outputs12, outputs13, outputs14}, {labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10, labels11, labels12, labels13, labels14})
		local gradOutputs = criterion:backward({outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, outputs12, outputs13, outputs14}, {labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10, labels11, labels12, labels13, labels14})
		model:backward(inputs, gradOutputs)
		return err, gradParameters

	end
	optim.sgd(feval, parameters, optimState)

	-- DataParallelTabel's syncParameters
	if model.needsSync then
		model:syncParameters()
	end

	cutorch.synchronize()
	batchNumber = batchNumber + 1
	loss_epoch = loss_epoch + err

	--print(('Ep. [%d/%d][%d/%d]\tTime %.3f Err %.5f LR %.0e DLTime %.3f'):format(epoch, opt.nEpochs, batchNumber, opt.epochSize, timer:time().real, err, optimState.learningRate, dataLoadingTime))

	dataTimer:reset()

end








