function compute_distance_joint (dataset, nJoints) 
	local pred_save = torch.Tensor(dataset.label:size(1), dataset.label:size(2))
	local dist_joints = torch.zeros(nJoints)
	for i=1, dataset.label:size(1) do
		local gt = dataset.label[i]
		local pred = model:forward(dataset.data[i])
		pred_save[{i,{}}] = pred:double()
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
		local MSE_each = 0
		for j = 1,dataset.label:size(2) do
			local diff = gt[j] - pred[j]
			MSE_each = MSE_each + math.pow(diff,2)
		end
		MSE = MSE + MSE_each
	end
	local avgMSE = MSE / dataset.label:size(1)

	return avgMSE	
end


function compute_PCP(dataset)

	local alpha = 0.5
	local nJoints = dataset.label:size(2)/2
	local nParts

	for iSmp = 1, dataset.label:size(1) do 			-- iterate through data samples

		local pred = model:forward(dataset.data[iSmp])
		local gt = dataset.label[iSmp]

		-- Case1: fullbody	
		if nJoints == 14 then
			nParts = 11

			jidx_part = torch.Tensor(nParts,2)
			jidx_part[1] = {1,2}
			jidx_part[2] = {2,3}
			jidx_part[3] = {3,4}
			jidx_part[4] = {4,5}
			jidx_part[5] = {2,9}
			jidx_part[6] = {9,10}
			jidx_part[7] = {10,11}
			jidx_part[8] = {6,7}
			jidx_part[9] = {7,8}
			jidx_part[10] = {12,13}
			jidx_part[11] = {13,14}

			local pcp_cnt = 0
			for iParts = 1, nParts do
				local e1 = {pred[2*jidx_part[iParts][1]-1], pred[2*jidx_part[iParts][1]]}
				local e2 = {pred[2*jidx_part[iParts][2]-1], pred[2*jidx_part[iParts][2]]}
				local g1 = {gt[2*jidx_part[iParts][1]-1], gt[2*jidx_part[iParts][1]]}
				local g2 = {gt[2*jidx_part[iParts][2]-1], gt[2*jidx_part[iParts][2]]}
				local dist_g1g2 = math.sqrt(math.pow(g1[1]-g2[1],2)+math.pow(g1[2]-g2[2],2)) 
				local dist_e1g1 = math.sqrt(math.pow(e1[1]-g1[1],2)+math.pow(e1[2]-g1[2],2))
				local dist_e2g2 = math.sqrt(math.pow(e2[1]-g2[1],2)+math.pow(e2[2]-g2[2],2))
				local dist_e1g2 = math.sqrt(math.pow(e1[1]-g2[1],2)+math.pow(e1[2]-g2[2],2))
				local dist_e2g1 = math.sqrt(math.pow(e2[1]-g1[1],2)+math.pow(e2[2]-g1[2],2))
				local L = dist_gig2

				-- check if pass the rule
				if ( (dist_e1g1 <= alpha*L and dist_e2g2 <= alpha*L) 
					or (dist_e1g2 <= alpha*L and dist_e2g1 <= alpha*L) ) then 
					pcp_cnt = pcp_cnt + 1
				end
			end
			print(pcp_cnt)

		-- Case2: upperbody
		elseif nJoints == 8 then
			nParts = 7


		-- Case3: lowerbody
		elseif nJoints == 6 then
			nParts = 4
		end



	end
	


end
