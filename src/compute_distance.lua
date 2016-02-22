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

function convert_multi_nofilt_label(label)
	-- label is 14 joint structured output label (as a table) 
	assert(type(label)=='table' and label[1]:size(1) == 2)
	local labelout = torch.Tensor(28):cuda()
	for i=1,14 do
		labelout[i*2-1] = label[i][1]
		labelout[i*2] = label[i][2]
	end
	return labelout	
end

function convert_torsolimbs_label(pred)
	-- Assumption: PR_multi (2 table) -->  PR_full (28 size tensor)
	-- torso and limbs 2 outputs
	assert(pred[1]:size(1) == 12 and pred[2]:size(1) == 16)
	
	local pred_tmp = torch.Tensor(2*nJoints)
	pred_tmp[1] = pred[1][1]
	pred_tmp[2] = pred[1][2]
	pred_tmp[3] = pred[1][3]
	pred_tmp[4] = pred[1][4]
	pred_tmp[5] = pred[1][5]
	pred_tmp[6] = pred[1][6]
	pred_tmp[11] = pred[1][7]
	pred_tmp[12] = pred[1][8]
	pred_tmp[17] = pred[1][9]
	pred_tmp[18] = pred[1][10]
	pred_tmp[23] = pred[1][11]
	pred_tmp[24] = pred[1][12]

	pred_tmp[7] = pred[2][1]
	pred_tmp[8] = pred[2][2]
	pred_tmp[9] = pred[2][3]
	pred_tmp[10] = pred[2][4]
	pred_tmp[13] = pred[2][5]
	pred_tmp[14] = pred[2][6]
	pred_tmp[15] = pred[2][7]
	pred_tmp[16] = pred[2][8]
	pred_tmp[19] = pred[2][9]
	pred_tmp[20] = pred[2][10]
	pred_tmp[21] = pred[2][11]
	pred_tmp[22] = pred[2][12]
	pred_tmp[25] = pred[2][13]
	pred_tmp[26] = pred[2][14]
	pred_tmp[27] = pred[2][15]
	pred_tmp[28] = pred[2][16]
	return pred_tmp
end

function convert_multi_label(pred)
	-- Assumption: PR_multi (2 table) -->  PR_full (28 size tensor)
	assert(pred[1]:size(1) == opt.batchSize and pred[2]:size(1) == opt.batchSize)

	local pred_tmp = torch.Tensor(opt.batchSize, 2*nJoints)
	for i=1,opt.batchSize do
		pred_tmp[i][1] = pred[1][i][1]
		pred_tmp[i][2] = pred[1][i][2]
		pred_tmp[i][3] = pred[1][i][3]
		pred_tmp[i][4] = pred[1][i][4]
		pred_tmp[i][5] = pred[1][i][5]
		pred_tmp[i][6] = pred[1][i][6]
		pred_tmp[i][7] = pred[1][i][7]
		pred_tmp[i][8] = pred[1][i][8]
		pred_tmp[i][9] = pred[1][i][9]
		pred_tmp[i][10] = pred[1][i][10]

		pred_tmp[i][17] = pred[1][i][11]
		pred_tmp[i][18] = pred[1][i][12]
		pred_tmp[i][19] = pred[1][i][13]
		pred_tmp[i][20] = pred[1][i][14]
		pred_tmp[i][21] = pred[1][i][15]
		pred_tmp[i][22] = pred[1][i][16]

		pred_tmp[i][11] = pred[2][i][1]
		pred_tmp[i][12] = pred[2][i][2]
		pred_tmp[i][13] = pred[2][i][3]
		pred_tmp[i][14] = pred[2][i][4]
		pred_tmp[i][15] = pred[2][i][5]
		pred_tmp[i][16] = pred[2][i][6]
			
		pred_tmp[i][23] = pred[2][i][7]
		pred_tmp[i][24] = pred[2][i][8]
		pred_tmp[i][25] = pred[2][i][9]
		pred_tmp[i][26] = pred[2][i][10]
		pred_tmp[i][27] = pred[2][i][11]
		pred_tmp[i][28] = pred[2][i][12]
	end
	return pred_tmp
end

function convert_fcnlabel (label)
	local label_new = torch.Tensor(2*nJoints)

	for i =1, nJoints do
		-- find max
		local count =0
		local idx_max = {}
		local max = torch.max(label[i])
		for j=1,label[i]:size(1) do
			for k=1,label[i]:size(2) do
				if label[i][j][k] == max then
					count = count + 1
					idx_max = {j,k}
				end
			end
		end

		-- new label. 
		label_new[2*i-1] = idx_max[2] / 16
		label_new[2*i] = idx_max[1] / 32
	end

	return label_new
end

function compute_epj (label_gt, label_pred) 
	assert(label_gt:size(1) == label_pred:size(1))

	local dist_joints = torch.zeros(nJoints)
	for i=1, label_pred:size(1) do
		local gt = label_gt[i]
		local pred = label_pred[i]

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
	local EPJ = dist_total/label_pred:size(1)
	local EPJ_avg = dist_total/label_pred:size(1)/nJoints
	
	return EPJ, EPJ_avg
end

function compute_MSE (label_gt, label_pred) 

	local MSE = 0
	for i = 1, label_pred:size(1) do
		local gt = label_gt[i]
		local pred = label_pred[i]

		-- compute MSE
		local MSE_each = 0
		for j = 1,nJoints*2 do
			local diff = gt[j] - pred[j]
			MSE_each = MSE_each + math.pow(diff,2)
		end
		MSE = MSE + MSE_each
	end
	local MSE_avg = MSE / label_pred:size(1)

	return MSE_avg	
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

function compute_PCP(label_gt, label_pred)

	assert(label_gt:size(1) == label_pred:size(1))

	local alpha = 0.5
	local nParts
	local pcp_cnt = 0

	for iSmp = 1, label_pred:size(1) do 			-- iterate through data samples
		
		local pred = label_pred[iSmp]
		local gt = label_gt[iSmp]
		local pcp_cnt_smp = 0

		-- Case1: fullbody	
		if opt.t == 'PR_full' or opt.t == 'PR_multi' or opt.t == 'PR_multi_test' or opt.t == 'PR_filt_struct' or opt.t == 'PR_eachjoint' or opt.t == 'PR_fcn' or opt.t == 'PR_torsolimbs' then
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
	
	PCP = ( pcp_cnt / (label_pred:size(1) * nParts) ) * 100

	return PCP

end

function compute_PCK(label_gt, label_pred)

	assert(label_gt:size(1) == label_pred:size(1))

	local alpha = 0.5
	local pck_cnt = 0

	for iSmp = 1, label_pred:size(1) do 			-- iterate through data samples
		
		local pred = label_pred[iSmp]
		local gt = label_gt[iSmp]

		-- compute size of head
		local hSize = math.sqrt(math.pow(gt[1]-gt[3],2)+math.pow(gt[2]-gt[4],2))

		-- if distance is less than alpha * head size then passed!!
		for i=1,14 do
			local d=math.sqrt(math.pow(gt[2*i-1]-pred[2*i-1],2)+math.pow(gt[2*i]-pred[2*i],2)) 
			if d <= hSize*alpha then
				pck_cnt = pck_cnt + 1
			end
		end

	end
	
	local PCK = ( pck_cnt / (nJoints * label_pred:size(1)) ) * 100

	return PCK

end

