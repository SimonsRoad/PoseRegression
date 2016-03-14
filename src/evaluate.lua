--[[ 
-- evaluate.lua
-- Namhoon Lee 
-- The Robotics Institute, Carnegie Mellon University
-- namhoonl@andrew.cmu.edu
--]]

local matio = require 'matio'



local function processBatch(inputsCPU, labelsCPU)
	local inputs = torch.CudaTensor()
	local labels = torch.CudaTensor()
	inputs:resize(inputsCPU:size()):copy(inputsCPU)
	labels:resize(labelsCPU:size()):copy(labelsCPU)

	local pred = model:forward(inputs)
	local gt = labelsCPU

	local pred_new = torch.Tensor(opt.batchSize, nJoints*2):cuda()
	local gt_new   = torch.Tensor(opt.batchSize, nJoints*2):cuda()

	-- resize pred
	if type(pred) == 'table' then
		if table.getn(pred)==2 and opt.t=='PR_multi' then           -- structured & no filter
			pred_new = convert_multi_label(pred)
			gt_new = gt
		elseif table.getn(pred)==2 and opt.t=='PR_torsolimbs' then
			assert(false, 'no implemented correctly yet')
			for i=1,opt.batchSize do
				pred_new[i] = convert_torsolimbs_label(pred[i])
			end
		elseif table.getn(pred) == 14 then
			assert(false, 'no implemented correctly yet')
			if pred:size(3) == 2 then        -- structured & no filter & each joint
				for i=1,opt.batchSize do
					pred_new[i] = convert_multi_nofilt_label(pred[i])
				end
			elseif pred:size(3) == 192 then  -- structured & filter
				for i=1,opt.batchSize do
					gt_new[i] = convert_filt_label(gt_new[i])
					pred_new[i] = convert_multi_filt_label(pred[i])
				end
			end
		end
	else
		-- case 3: fcn label
		-- convert
		for i=1,pred:size(1) do
			pred_new[i] = convert_fcnlabel(pred[i])
			gt_new[i]   = convert_fcnlabel(gt[i])
		end
	end

	-- At this stage, the size of lable should be 28
	assert(gt_new:size(1) == opt.batchSize)
	assert(gt_new:size(2) == 2*nJoints)
	assert(pred_new:size(1) == opt.batchSize)
	assert(pred_new:size(2) == 2*nJoints)

	--return gt_new, pred_new, gt_fcn, pred_fcn
	return gt_new, pred_new

end

local function forwardpass(dataset)
	local nData = dataset.label:size(1)
	local gt	= torch.Tensor(nData, nJoints*2):cuda()
	local pred 	= torch.Tensor(nData, nJoints*2):cuda()

	for i=1, math.ceil(nData/opt.batchSize) do
		local idx_start = (i-1) * opt.batchSize + 1
		local idx_end   = idx_start + opt.batchSize - 1
		local idx_batch
		if idx_end <= nData then
			idx_batch = torch.range(idx_start, idx_end)
		else
			local idx1 = torch.range(idx_start, nData)
			local idx2 = torch.range(1, idx_end-nData)
			idx_batch = torch.cat(idx1, idx2, 1)
		end
		
		local inputs, labels
		inputs = dataset.data:index(1, idx_batch:long())
		labels = dataset.label:index(1, idx_batch:long())

		-- process batch
		local gt_batch, pred_batch = processBatch(inputs, labels)

        print(gt_batch:size())
        adf=adf+1

		if opt.t == 'PR_multi' then
			if idx_end <= nData then
				gt[{ {idx_start,idx_end}, {} }]				= gt_batch
				pred[{ {idx_start,idx_end}, {}}]	 		= pred_batch
			else 
				local len = nData-idx_start+1
				gt[{ {idx_start,nData}, {} }]			= gt_batch[{{1,len},{}}]
				pred[{ {idx_start,nData}, {}}]	 		= pred_batch[{{1,len},{}}]
			end
		elseif opt.t == 'PR_fcn' then
			if idx_end <= nData then
				gt[{ {idx_start,idx_end}, {} }]				= gt_batch
				pred[{ {idx_start,idx_end}, {}}]	 		= pred_batch
			else 
				local len = nData-idx_start+1
				gt[{ {idx_start,nData}, {} }]			= gt_batch[{{1,len},{}}]
				pred[{ {idx_start,nData}, {}}]	 		= pred_batch[{{1,len},{}}]
			end
		end
	
	end

	return gt, pred
end

function evaluate(dataset, kind, savedir)

	-- 0. forward pass and convert labels to single vectors
	-- labels are all #Data x 28
	label_gt, label_pred = forwardpass(dataset)
	

	-- EVALUATE
	PCP = compute_PCP(label_gt, label_pred)
	PCK = compute_PCK(label_gt, label_pred)
	--EPJ, EPJ_avg = compute_epj(label_gt, label_pred)
	--MSE_avg = compute_MSE(label_gt, label_pred)

	-- print out the results
	print(string.format('-- (%s)', kind))
	print(string.format('PCP     :   %.2f  (%%)', PCP))
	print(string.format('PCK     :   %.2f  (%%)', PCK))
	--print(string.format('EPJ_avg :   %.4f', EPJ_avg))
	--print(string.format('MSE_avg :   %.4f', MSE_avg))

	-- save into log
	if kind == 'test' then
		pckLogger_test:add{ ['PCK'] = PCK }
	elseif kind == 'train' then
		pckLogger_train:add{ ['PCK'] = PCK }
	end

	-- save prediction results for visualization
	pred_save = label_pred:float()
	if savedir then
		-- be careful when saving
		adf=adf+1

		-- This is when temporary save directory is specified
		matio.save(string.format('../save/%s/pred_%s_%s.mat', savedir, kind, opt.t), pred_save)
		--matio.save(string.format('../save/%s/pred_%s_%s_heatmap.mat', savedir, kind, opt.t), label_pred_fcn)
		--matio.save(string.format('../save/%s/gt_%s_%s_heatmap.mat', savedir, kind, opt.t), label_gt_fcn)
	else
		matio.save(paths.concat(opt.save,string.format('pred_%s_%s.mat', kind, opt.t)), pred_save)
		--matio.save(paths.concat(opt.save,string.format('pred_%s_%s_heatmap.mat', kind, opt.t)), label_pred_fcn)
		--matio.save(paths.concat(opt.save,string.format('gt_%s_%s_heatmap.mat', kind, opt.t)), label_gt_fcn)
	end
	--matio.save(string.format('pred_%s_%s.mat', kind, opt.t), pred_save)
	
end




