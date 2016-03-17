--[[
-- train.lua
-- This code is from imigenet-multiGPU.torch by soumith
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


trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local loss_epoch
local batchNumber 


function train()

	cutorch.synchronize()

	-- set the dropouts to training mode
	model:training()

	local tm = torch.Timer()
	loss_epoch = 0
    batchNumber = 0

	-- randomize dataset (actually just indices)
	local idx_rand = torch.randperm(opt.nTrainData)

    -- training
	for i=1,opt.epochSize do
		-- create indices for a batch
		local idx_start = (i-1)*opt.batchSize + 1
		local idx_end   = idx_start + opt.batchSize - 1
		local idx_batch
		if idx_end <= opt.nTrainData then
			idx_batch = idx_rand[{{idx_start,idx_end}}] 
		else 
			local idx1 = idx_rand[{{idx_start,opt.nTrainData}}]
			local idx2 = idx_rand[{{1,idx_end-opt.nTrainData}}]
			idx_batch = torch.cat(idx1, idx2, 1)
		end


        donkeys:addjob(
            function()
                local trainset_batch = loader:load_batch(idx_batch)
                return trainset_batch.data, trainset_batch.label
            end,
		    trainBatch
        )
	end

    donkeys:synchronize()
	cutorch.synchronize()

	loss_epoch = loss_epoch / opt.epochSize

	trainLogger:add{
		['avg loss (train set)'] = loss_epoch
	}
	print(string.format('Ep. [%d/%d] (Train) Time(s): %.2f  ' .. 'avg loss (per batch): %.8f ', epoch, opt.nEpochs, tm:time().real, loss_epoch))


	collectgarbage()

	if epoch % 1 == 0 then
        if torch.type(model) == 'nn.DataParallelTable' then
            model = model:get(1)
        end
        torch.save(paths.concat(opt.save, 'model_'..epoch..'.t7'), model)
        torch.save(paths.concat(opt.save, 'optimState_'..epoch..'.t7'), optimState)
	end

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
		err = criterion:forward(outputs, labels)
		local gradOutputs = criterion:backward(outputs, labels)
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

    print(string.format('Ep. [%d/%d][%d/%d] Time(s): %.2f  ' .. 'batch err: %.7f | dataLoadingTime: %.3f', epoch, opt.nEpochs, batchNumber, opt.epochSize, timer:time().real, err, dataLoadingTime))
    dataTimer:reset()

end


