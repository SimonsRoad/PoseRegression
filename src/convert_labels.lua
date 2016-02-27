-- [[
-- convert_labels.lua
-- Namhoon Lee (namhoonl@andrew.cmu.edu), The Robotics Institute, Carnegie Mellon Univ.
--
-- 1) convert 28 to 14*(64+128)
-- 2) convert 14*(64+128) to 28
--
-- ]]
local matio = require 'matio'

local function filter_gaussian(label, maxPixelSize, stdv)
	local label_new = torch.FloatTensor(label:size(1), label:size(2), maxPixelSize):zero()
	local pixels = torch.range(1,maxPixelSize):float()

	local k = 0

	for i = 1, label:size(1) do
		local keypoints = label[i]
		local new_keypoints = label_new[i]
		for j = 1, label:size(2) do
			local kp = keypoints[j]
			if kp ~= -1 then
				local new_kp = new_keypoints[j]
				new_kp:add(pixels, -kp)
				new_kp:cmul(new_kp)
				new_kp:div(2*stdv*stdv)
				new_kp:mul(-1)
				new_kp:exp(new_kp)
				new_kp:div(math.sqrt(2*math.pi)*stdv)
			else
				k = k + 1
			end
		end
	end

	return label_new

end

function convert_labels_to_spatialLabels(label_ori)

	-- scale back to 64*128
	local w = 64
	local h = 128
	local tmp = torch.repeatTensor(torch.Tensor({w,h}),label_ori:size(1),14)
	local label_res = torch.cmul(label_ori, tmp):round()

	-- split x and y
	local label_x = label_res:index(2, torch.LongTensor{1,3,5,7,9,11,13,15,17,19,21,23,25,27})
	local label_y = label_res:index(2, torch.LongTensor{2,4,6,8,10,12,14,16,18,20,22,24,26,28})

	-- Guassian filter; perform twice for x and y respectively
	local label_x_filt = filter_gaussian(label_x, w, 0.8)
	local label_y_filt = filter_gaussian(label_y, h, 0.8)

	-- concatenate x and y
	local label_xy = torch.cat(label_x_filt, label_y_filt, 3)

	-- reshape as a single vector
	local label = torch.reshape(label_xy, label_xy:size(1), label_xy:size(2)*label_xy:size(3))

	-- test1
	--print(label_xy[1][1][30])
	--print(label[1][30][1])
	-- test2
	--label_tmp = torch.reshape(label, label:size(1), label_xy:size(2), label_xy:size(3))
	--print(label_xy[1][1][30])
	--print(label_tmp[1][1][30])
	--print(label_tmp:size())

	return label

end


function convert_spatialLabels_to_labels(label)
	-- The purpose of this function is to restore the joint locations from spatial labels
	-- I attempt to do this by finding out maximum.
	-- There should be space for improvement!
	
	-- reshape
	local label_res = torch.reshape(label, label:size(1), 14, (64+128))

	-- split x and y
	local label_x = label_res[{ {}, {}, {1,64} }]
	local label_y = label_res[{ {}, {}, {65,(64+128)} }]
	assert(label_y:size(2) == 14 and label_y:size(3) == 128)

	-- 
	local label_joint = torch.Tensor(label:size(1), 14*2)
	for i=1,label:size(1) do
		-- find out joint location. (Idealy, IFFT should be performed)
		local _, idx_max_x = torch.max(label_x[{ {i}, {}, {} }], 3)
		local _, idx_max_y = torch.max(label_y[{ {i}, {}, {} }], 3)
		idx_max_x = torch.reshape(idx_max_x, idx_max_x:nElement()):float()
		idx_max_y = torch.reshape(idx_max_y, idx_max_y:nElement()):float()

		-- normalize it so that it ranges between 0 and 1
		idx_max_x:div(64)
		idx_max_y:div(128)
		assert(idx_max_y:size(1) == 14)

		-- rearrange to [x1,y1, x2,y2, ... , x14,y14]
		local label_joint_smp = torch.cat(idx_max_x, idx_max_y, 2)
		label_joint_smp = torch.reshape(label_joint_smp, 28)

		label_joint[i] = label_joint_smp
	end

	return label_joint
end

-- 
function convert_labels_to_fcnLabels(labels)
	-- converts the normal label to FCN voxel
	-- input label: (#data) x 28
	-- output label: (#data) x 14 x 32 x 16 
	-- 14: # of joints
	-- 32: height
	-- 16: width
	
	-- 00. setting
	local nlabels = labels:size(1)
	local w = 16
	local h = 32

	local grid_x = torch.repeatTensor(torch.range(1,16),32,1)
	local grid_y = torch.repeatTensor(torch.range(1,32),16,1):transpose(1,2)
	local sigma = 1.0


	-- 0. convert normalized labels to pixel labels
	local scalar = torch.repeatTensor(torch.Tensor({w,h}), nlabels, nJoints):cuda()
	local plabels = torch.cmul(labels, scalar)


	-- 1. reshape labels and apply Gaussian filter: output as voxels
	local fcnlabels = torch.Tensor(nlabels, nJoints, h, w)
	for i=1,nlabels do

		local voxel = torch.Tensor(nJoints, h, w)
		for j=1,nJoints do
			local x = plabels[i][2*j-1]
			local y = plabels[i][2*j]
			local tmp1 = (1/(sigma*math.sqrt(2*math.pi)))
			local tmp2 = torch.exp(-( torch.pow(grid_x-x,2) + torch.pow(grid_y-y,2) )/(2*math.pow(sigma,2)))
			local channel = torch.mul(tmp2, tmp1)
			channel:div(torch.max(channel))
			voxel[j] = channel
		end
		fcnlabels[i] = voxel
	end

	
	return fcnlabels
end

















