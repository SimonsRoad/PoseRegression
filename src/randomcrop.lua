--[[
-- randomcrop.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]

function crop_label(img_ori, label_ori)

	-- bw_outer, bh_outer
	local bw_outer = 90
	local bh_outer = bw_outer*2
	local bw_crop = 64
	local bh_crop = 128

	-- compute bw_tight, bh_tight
	local label_res = torch.reshape(label_ori, 14, 2)
	local label_x_min = torch.min(label_res[{ {}, {1}}])
	local label_x_max = torch.max(label_res[{ {}, {1}}])
	local label_y_min = torch.min(label_res[{ {}, {2}}])
	local label_y_max = torch.max(label_res[{ {}, {2}}])

	-- lt_tight, rb_tight
	local lt_tight_x = label_x_min * bw_outer
	local lt_tight_y = label_y_min * bh_outer
	local rb_tight_x = label_x_max * bw_outer
	local rb_tight_y = label_y_max * bh_outer
	local bw_tight = rb_tight_x - lt_tight_x
	local bh_tight = rb_tight_y - lt_tight_y

	-- lt_crop - center
	local d_from_tight_to_centeredbox_x = (64-bw_tight)/2 
	local d_from_tight_to_centeredbox_y = (128-bh_tight)/2
	local lt_crop_x_center = lt_tight_x - d_from_tight_to_centeredbox_x
	local lt_crop_y_center = lt_tight_y - d_from_tight_to_centeredbox_y

	-- lt_crop - min 
	local lt_crop_x_min = lt_crop_x_center - 4
	local lt_crop_y_min = lt_crop_y_center - 8
	
	-- lt_crop - max
	local lt_crop_x_max = lt_crop_x_center + 4
	local lt_crop_y_max = lt_crop_y_center + 8
	--print(lt_crop_x_min, lt_crop_y_min)
	--print(lt_crop_x_max, lt_crop_y_max)
	
	if math.floor(lt_crop_x_min) <= 0 then
		lt_crop_x_min = 1
	end
	if math.ceil(lt_crop_x_max)+ 64 >= bw_outer then
		lt_crop_x_max = bw_outer-64-1
	end

	-- randomly select it
	local lt_x_sel = math.ceil(torch.uniform(lt_crop_x_min, lt_crop_x_max))
	local lt_y_sel = math.ceil(torch.uniform(lt_crop_y_min, lt_crop_y_max))

	-- local image, resize and crop
	local img_out = image.scale(img_ori, bw_outer .. 'x' .. bh_outer)
	img_out = image.crop(img_out, lt_x_sel, lt_y_sel, lt_x_sel+bw_crop, lt_y_sel+bh_crop)
	assert(img_out:size(2) == bh_crop and img_out:size(3) == bw_crop)

	-- transform label
	local scalar = torch.repeatTensor(torch.Tensor({bw_outer, bh_outer}), 14, 1)
	local label_out = torch.cmul(label_res, scalar)
	local translation = torch.repeatTensor(torch.Tensor({lt_x_sel, lt_y_sel}), 14)
	label_out:add(-translation)
	scalar = torch.repeatTensor(torch.Tensor({bw_crop, bh_crop}), 14, 1)
	label_out:cdiv(scalar)
	label_out = torch.reshape(label_out, 28)

	-- sanity check
	for i=1, label_out:size(1) do
		if label_out[i] <= 0.0 or label_out[i] >= 1.0 then
			print(label_out[i])
			assert(false, 'this should not happen!')
		end
	end

	return img_out, label_out
end



function randomcrop(dataset)
	assert(dataset.label:size(1) == dataset.data:size(1))

	local nData = dataset.label:size(1)

	local imagetensor = torch.Tensor(nData, 3, 128, 64)
	local labeltensor = torch.Tensor(nData, 28)

	--[[  Debugging purpose
	local tmptensor = torch.Tensor(nData,1)
	for i=1,nData do
		tmptensor[i] = crop_label(dataset.data[i], dataset.label[i])
	end
	print(torch.min(tmptensor))
	print(torch.max(tmptensor))
	adf=adf+1
	--]]
	
	for i=1,nData do
		local crop, label = crop_label(dataset.data[i], dataset.label[i])
		imagetensor[i] = crop
		labeltensor[i] = label
	end

	local out = {data = imagetensor, label = labeltensor}
	return out
end
