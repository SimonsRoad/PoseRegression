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

	local pred_new = torch.Tensor(opt.batchSize, 28)
	local gt_new   = torch.Tensor(opt.batchSize, 28)
	local gt_fcn, pred_fcn

	-- resize pred
	if type(pred) == 'table' then
		if table.getn(pred)==2 and opt.t=='PR_multi' then           -- structured & no filter
			pred_new = convert_multi_label(pred)
		elseif table.getn(pred)==2 and opt.t=='PR_torsolimbs' then
			for i=1,opt.batchSize do
				pred_new[i] = convert_torsolimbs_label(pred[i])
			end
		elseif table.getn(pred) == 14 then
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
		if pred:size(2) == nJoints and pred:size(3) == 32 and pred:size(4) == 16 then
			-- before converting, save the heatmap
			gt_fcn   = gt:float()
			pred_fcn = pred:float()

			-- convert
			for i=1,pred:size(1) do
				pred_new[i] = convert_fcnlabel(pred[i])
				gt_new[i]   = convert_fcnlabel(gt[i])
			end
		end

	end

		-- At this stage, the size of lable should be 28
	assert(gt_new:size(1) == opt.batchSize)
	assert(gt_new:size(2) == 2*nJoints)
	assert(pred_new:size(1) == opt.batchSize)
	assert(pred_new:size(2) == 2*nJoints)

	return gt_new, pred_new, gt_fcn, pred_fcn

end

local function forwardpass(inputdataset)
	local nData = inputdataset.label:size(1)
	local gt		= torch.Tensor(nData, 28):float()
	local pred 		= torch.Tensor(nData, 28)
	local gt_fcn  	= torch.Tensor(nData, 14, 32, 16)	--fcn labels to see heatmap
	local pred_fcn	= torch.Tensor(nData, 14, 32, 16)	--fcn labels to see heatmap

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
		inputs = inputdataset.data:index(1, idx_batch:long())
		labels = inputdataset.label:index(1, idx_batch:long())

		-- process batch
		local gt_batch, pred_batch, gt_fcn_batch, pred_fcn_batch = processBatch(inputs, labels)

		
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
				gt_fcn[{ {idx_start,idx_end},{},{},{}}] 	= gt_fcn_batch
				pred_fcn[{ {idx_start,idx_end},{},{},{}}] 	= pred_fcn_batch
			else 
				local len = nData-idx_start+1
				gt[{ {idx_start,nData}, {} }]			= gt_batch[{{1,len},{}}]
				pred[{ {idx_start,nData}, {}}]	 		= pred_batch[{{1,len},{}}]
				gt_fcn[{ {idx_start,nData},{},{},{}}] 	= gt_fcn_batch[{{1,len},{},{},{}}]
				pred_fcn[{ {idx_start,nData},{},{},{}}] = pred_fcn_batch[{{1,len},{},{},{}}]
			end
		end
	
	end

	return gt, pred, gt_fcn, pred_fcn

end


function evaluate(inputdataset, kind, savedir)

	-- 0. forward pass and convert labels to single vectors
	-- labels are all #Data x 28
	label_gt, label_pred, label_gt_fcn, label_pred_fcn = forwardpass(inputdataset)
	

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

	-- save prediction results for visualization
	pred_save = label_pred:double()
	if savedir then
		-- This is when temporary save directory is specified
		matio.save(string.format('../save/%s/pred_%s_%s.mat', savedir, kind, opt.t), pred_save)
		matio.save(string.format('../save/%s/pred_%s_%s_heatmap.mat', savedir, kind, opt.t), label_pred_fcn)
		matio.save(string.format('../save/%s/gt_%s_%s_heatmap.mat', savedir, kind, opt.t), label_gt_fcn)
	else
		matio.save(paths.concat(opt.save,string.format('pred_%s_%s.mat', kind, opt.t)), pred_save)
		matio.save(paths.concat(opt.save,string.format('pred_%s_%s_heatmap.mat', kind, opt.t)), label_pred_fcn)
		matio.save(paths.concat(opt.save,string.format('gt_%s_%s_heatmap.mat', kind, opt.t)), label_gt_fcn)
	end
	--matio.save(string.format('pred_%s_%s.mat', kind, opt.t), pred_save)
end




