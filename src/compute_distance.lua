function compute_distance (dataset, nJoints) 
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
	for i = 1, dataset.label(1) do
		local gt = dataset.label[i]
		local pred = model:forward(dataset.data[i])
		local MSE_each = 0
		for j = 1,dataset.label(2) do
			local diff = gt[j] - pred[j]
			MSE_each = MSE_each + math.pow(diff,2)
		end
		MSE = MSE + MSE_each
	end
	local avgMSE = MSE / dataset.label(1)

	return avgMSE	
end



