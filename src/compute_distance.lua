function convert_multi_filt_label(label)
	-- Assume the input label is a TABLE
	assert(table.getn(label)==14)
	
	local label_new = torch.Tensor(2*nJoints):cuda()
	for i = 1, nJoints do
		local label_joint = label[i]
		local label_x = label_joint[{{1,W}}]
		local label_y = label_joint[{{W+1,W+H}}]
		
		local _, idx_max_x = torch.max(label_x,1)
		local _, idx_max_y = torch.max(label_y,1)

		idx_max_x:float()
		idx_max_y:float()
		idx_max_x:div(W)
		idx_max_y:div(H)

		label_new[(2*i)-1] = idx_max_x
		label_new[(2*i)] = idx_max_y
	end
	
	return label_new

end

function convert_filt_label(label)
	-- here input argument is a filtered 'long' label
	-- This should be actually inverse filtering.. (iFFT can do this..)

	-- reshape 
	local label_res = torch.reshape(label, nJoints, (W+H))

	-- split x and y
	local label_x = label_res[{ {}, {1,W}}]
	local label_y = label_res[{ {}, {W+1,(W+H)} }]
	assert(label_y:size(1) == nJoints and label_y:size(2) == H)

	-- find out joint location. (Iealy, IFFT should be performed)
	local _, idx_max_x = torch.max(label_x,2)
	local _, idx_max_y = torch.max(label_y,2)
	idx_max_x:float()
	idx_max_y:float()
	idx_max_x:div(W)
	idx_max_y:div(H)
	--print(idx_max_x)
	--print(idx_max_y)
	assert(idx_max_y:size(1) == nJoints)

	-- rearrange to [x1,y1,x2,y2, ... , x14,y14]
	local label_joint = torch.cat(idx_max_x, idx_max_y, 2)
	label_joint = torch.reshape(label_joint, 2*nJoints)

	return label_joint 
end


function convert_multi_label(pred)
	-- Assumption: PR_multi (2 table) -->  PR_full (28 size tensor)
	assert(pred[1]:size(1) == 16 and pred[2]:size(1) == 12)
	
	local pred_tmp = torch.Tensor(2*nJoints)
	pred_tmp[1] = pred[1][1]
	pred_tmp[2] = pred[1][2]
	pred_tmp[3] = pred[1][3]
	pred_tmp[4] = pred[1][4]
	pred_tmp[5] = pred[1][5]
	pred_tmp[6] = pred[1][6]
	pred_tmp[7] = pred[1][7]
	pred_tmp[8] = pred[1][8]
	pred_tmp[9] = pred[1][9]
	pred_tmp[10] = pred[1][10]

	pred_tmp[17] = pred[1][11]
	pred_tmp[18] = pred[1][12]
	pred_tmp[19] = pred[1][13]
	pred_tmp[20] = pred[1][14]
	pred_tmp[21] = pred[1][15]
	pred_tmp[22] = pred[1][16]

	pred_tmp[11] = pred[2][1]
	pred_tmp[12] = pred[2][2]
	pred_tmp[13] = pred[2][3]
	pred_tmp[14] = pred[2][4]
	pred_tmp[15] = pred[2][5]
	pred_tmp[16] = pred[2][6]
			
	pred_tmp[23] = pred[2][7]
	pred_tmp[24] = pred[2][8]
	pred_tmp[25] = pred[2][9]
	pred_tmp[26] = pred[2][10]
	pred_tmp[27] = pred[2][11]
	pred_tmp[28] = pred[2][12]
	return pred_tmp
end


function compute_distance_joint (dataset, nJoints) 
	local pred_save = torch.Tensor(dataset.label:size(1), nJoints*2)
	local dist_joints = torch.zeros(nJoints)
	for i=1, dataset.label:size(1) do
		local gt = dataset.label[i]
		local pred = model:forward(dataset.data[i])

		-- resize pred
		if type(pred) == 'table' then
			if table.getn(pred) == 2 then			-- structured & no filter
				pred = convert_multi_label(pred)
			elseif table.getn(pred) == 14 then		-- structured & filter
				gt = convert_filt_label(gt)
				pred = convert_multi_filt_label(pred)
			end
		end

		-- case 2: a long filtered label
		if pred:size(1) == LLABEL and gt:size(1) == LLABEL then
			assert(LLABEL == 14*(64+128))
			pred = convert_filt_label(pred)
			gt = convert_filt_label(gt)
		end

		-- At this stage, the size of lable should be 28
		assert(pred:size(1) == 2*nJoints)

		-- save prediction for later use
		pred_save[{i,{}}] = pred:double()

		-- compute joint distance
		for j=1,nJoints do
			local xdiff = gt[2*j-1] - pred[2*j-1]
			local ydiff = gt[2*j] - pred[2*j]
			dist_joints[j] = dist_joints[j] + math.sqrt(math.pow(xdiff,2)+math.pow(ydiff,2))
		end
	end

	local dist_total = 0
	for i=1,nJoints do
		dist_total = dist_total + dist_joints[i]
	end
	errPerJoint = dist_total/dataset.label:size(1)
	meanErrPerJoint = dist_total/dataset.label:size(1)/nJoints
	
	return pred_save, errPerJoint, meanErrPerJoint
end

function compute_distance_MSE (dataset) 

	-- compute MSE 
	local MSE = 0
	for i = 1, dataset.label:size(1) do
		local gt = dataset.label[i]
		local pred = model:forward(dataset.data[i])

		-- resize pred
		if type(pred) == 'table' then
			if table.getn(pred) == 2 then			-- structured & no filter
				pred = convert_multi_label(pred)
			elseif table.getn(pred) == 14 then		-- structured & filter
				gt = convert_filt_label(gt)
				pred = convert_multi_filt_label(pred)
			end
		end

		-- case 2: a long filtered label
		if pred:size(1) == LLABEL and gt:size(1) == LLABEL then
			assert(LLABEL == 14*(64+128))
			pred = convert_filt_label(pred)
			gt = convert_filt_label(gt)
		end

		-- At this stage, the size of lable should be 28
		assert(pred:size(1) == 2*nJoints)

		-- compute MSE
		local MSE_each = 0
		for j = 1,nJoints*2 do
			local diff = gt[j] - pred[j]
			MSE_each = MSE_each + math.pow(diff,2)
		end
		MSE = MSE + MSE_each
	end
	local avgMSE = MSE / dataset.label:size(1)

	return avgMSE	
end


local function pcp_tester(jidx_part, iParts, pred, gt, alpha)
	local pcp = 0
	local e1 = torch.Tensor({pred[2*jidx_part[iParts][1]-1], pred[2*jidx_part[iParts][1]]})
	local e2 = torch.Tensor({pred[2*jidx_part[iParts][2]-1], pred[2*jidx_part[iParts][2]]})
	local g1 = torch.Tensor({gt[2*jidx_part[iParts][1]-1], gt[2*jidx_part[iParts][1]]})
	local g2 = torch.Tensor({gt[2*jidx_part[iParts][2]-1], gt[2*jidx_part[iParts][2]]})
	local dist_g1g2 = math.sqrt(math.pow(g1[1]-g2[1],2)+math.pow(g1[2]-g2[2],2)) 
	local dist_e1g1 = math.sqrt(math.pow(e1[1]-g1[1],2)+math.pow(e1[2]-g1[2],2))
	local dist_e2g2 = math.sqrt(math.pow(e2[1]-g2[1],2)+math.pow(e2[2]-g2[2],2))
	local dist_e1g2 = math.sqrt(math.pow(e1[1]-g2[1],2)+math.pow(e1[2]-g2[2],2))
	local dist_e2g1 = math.sqrt(math.pow(e2[1]-g1[1],2)+math.pow(e2[2]-g1[2],2))
	local L = dist_g1g2

	-- check if pass the rule
	if ( (dist_e1g1 <= alpha*L and dist_e2g2 <= alpha*L) 
		or (dist_e1g2 <= alpha*L and dist_e2g1 <= alpha*L) ) then 
		pcp = 1
	end
	return pcp
end

function compute_PCP(dataset)

	local alpha = 0.5
	local nParts
	local pcp_cnt = 0

	for iSmp = 1, dataset.label:size(1) do 			-- iterate through data samples

		local pred = model:forward(dataset.data[iSmp])
		local gt = dataset.label[iSmp]
		local pcp_cnt_smp = 0

		-- resize pred
		if type(pred) == 'table' then
			if table.getn(pred) == 2 then			-- structured & no filter
				pred = convert_multi_label(pred)
			elseif table.getn(pred) == 14 then		-- structured & filter
				gt = convert_filt_label(gt)
				pred = convert_multi_filt_label(pred)
			end
		end

		-- case 2: a long filtered label
		if pred:size(1) == LLABEL and gt:size(1) == LLABEL then
			assert(LLABEL == 14*(64+128))
			pred = convert_filt_label(pred)
			gt = convert_filt_label(gt)
		end

		-- At this stage, the size of lable should be 28
		assert(pred:size(1) == 2*nJoints)

		-- Case1: fullbody	
		if opt.t == 'PR_full' or opt.t == 'PR_multi' or opt.t == 'PR_multi_test' or opt.t == 'PR_filt' or opt.t == 'PR_filt_struct' then
			nParts = 11

			jidx_part = torch.Tensor(nParts,2)
			jidx_part[1] = torch.Tensor({1,2})
			jidx_part[2] = torch.Tensor({2,3})
			jidx_part[3] = torch.Tensor({3,4})
			jidx_part[4] = torch.Tensor({4,5})
			jidx_part[5] = torch.Tensor({2,9})
			jidx_part[6] = torch.Tensor({9,10})
			jidx_part[7] = torch.Tensor({10,11})
			jidx_part[8] = torch.Tensor({6,7})
			jidx_part[9] = torch.Tensor({7,8})
			jidx_part[10] = torch.Tensor({12,13})
			jidx_part[11] = torch.Tensor({13,14})

			for iParts = 1, nParts do
				pcp_cnt_smp = pcp_cnt_smp + pcp_tester(jidx_part,iParts,pred,gt,alpha)
			end

		-- Case2: upperbody
		elseif opt.t == 'PR_upper' then
			nParts = 7

			jidx_part = torch.Tensor(nParts,2)
			jidx_part[1] = torch.Tensor({1,2})
			jidx_part[2] = torch.Tensor({2,3})
			jidx_part[3] = torch.Tensor({3,4})
			jidx_part[4] = torch.Tensor({4,5})
			jidx_part[5] = torch.Tensor({2,6})
			jidx_part[6] = torch.Tensor({6,7})
			jidx_part[7] = torch.Tensor({7,8})

			for iParts = 1, nParts do
				pcp_cnt_smp = pcp_cnt_smp + pcp_tester(jidx_part,iParts,pred,gt,alpha)
			end

		-- Case3: lowerbody
		elseif opt.t == 'PR_lower' then
			nParts = 4

			jidx_part = torch.Tensor(nParts,2)
			jidx_part[1] = torch.Tensor({1,2})
			jidx_part[2] = torch.Tensor({2,3})
			jidx_part[3] = torch.Tensor({4,5})
			jidx_part[4] = torch.Tensor({5,6})

			for iParts = 1, nParts do
				pcp_cnt_smp = pcp_cnt_smp + pcp_tester(jidx_part,iParts,pred,gt,alpha)
			end
		end

		--print(string.format('PCP per sample: %.1f (%d/%d)',pcp_cnt_smp/nParts*100, pcp_cnt_smp, nParts))
		pcp_cnt = pcp_cnt + pcp_cnt_smp

	end
	
	PCP = ( pcp_cnt / (dataset.label:size(1) * nParts) ) * 100

	return PCP

end
