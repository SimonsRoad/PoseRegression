
function convert_multi_label(pred)
	-- Assumption: PR_multi (2 table) -->  PR_full (28 size tensor)
	assert(pred[1]:size(1) == opt.batchSize and pred[2]:size(1) == opt.batchSize)

	local pred_tmp = torch.Tensor(opt.batchSize, 2*nJoints):cuda()
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

	local label_new = torch.Tensor(2*nJoints):cuda()

	for i = 1, nJoints do
        print(i)
        adf=adf+1

		local max, idx = torch.max(torch.reshape(label[i], 128*64), 1)
		local j = math.floor(idx[1]/64)+1
		local k = idx[1] % 64

		-- new label. 
		label_new[2*i-1] = k / 64
		label_new[2*i] = j / 128
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

local function pcp_tester(jidx_part, iParts, pred, gt, alpha)
	local pcp = 0
	local e1 = torch.Tensor({pred[2*jidx_part[iParts][1]-1], pred[2*jidx_part[iParts][1]]})
	local e2 = torch.Tensor({pred[2*jidx_part[iParts][2]-1], pred[2*jidx_part[iParts][2]]})
	local g1 = torch.Tensor({gt[2*jidx_part[iParts][1]-1], gt[2*jidx_part[iParts][1]]})
	local g2 = torch.Tensor({gt[2*jidx_part[iParts][2]-1], gt[2*jidx_part[iParts][2]]})
	local scalar = torch.Tensor({64,128}) 
	e1 = torch.cmul(scalar, e1)
	e2 = torch.cmul(scalar, e2)
	g1 = torch.cmul(scalar, g1)
	g2 = torch.cmul(scalar, g2)
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
			nParts = 13

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
			jidx_part[12] = torch.Tensor({3,6})
			jidx_part[13] = torch.Tensor({9,12})

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
		local scalar = torch.repeatTensor(torch.Tensor({64,128}),14, 1):cuda()
		gt = torch.cmul(gt, scalar)
		pred = torch.cmul(pred, scalar)

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

