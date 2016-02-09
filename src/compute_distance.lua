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
