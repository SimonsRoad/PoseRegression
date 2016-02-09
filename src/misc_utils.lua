function get_time()
	local t = os.date("*t", os.time())
	local time = ('m'..t['month']..'d'..t['day']..'h'..t['hour']..'m'..t['min']..'s'..t['sec'])
	return time
end


function randomizeIndices(indices)
	local tmp = torch.randperm(indices:size(1))
	local out = torch.Tensor(indices:size(1))
	for i = 1, indices:size(1) do
		out[i] = indices[tmp[i]]
	end
	return out
end
