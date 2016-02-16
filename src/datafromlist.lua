-- datafromlist.lua

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
local matio = require 'matio'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

local sampleSize = {3, 128, 64}

local function loadImage(path)
   	local input = image.load(path, 3, 'float')   
   	-- resize
   	input = image.scale(input, '64x128') 
   	return input
end


local dataset = torch.class('dataLoader')
local initcheck = argcheck{
   pack=true,
   {name="filename",
    type="string",
    help="filenmae with list of images"}
}

function dataset:__init(...)
	local args =  initcheck(...)
	for k,v in pairs(args) do self[k] = v end
	self.numSamples = tonumber(sys.fexecute("cat " .. self.filename .. " |  wc -l"))
	self.maxFileLength = tonumber(sys.fexecute("cat " .. self.filename .. " |  awk '{print length($0)}' | datamash max 1"))
   	self.imagePath = torch.CharTensor() -- path to each image in dataset
	self.imagePath:resize(self.numSamples, self.maxFileLength)
   	local i_data = self.imagePath:data()
   	local file = assert(io.open(self.filename, "r"))
   	self.pathLength = torch.LongTensor(self.numSamples):fill(0)   
   	local count = 1
   	for line in file:lines() do
    	self.pathLength[count] = line:len()
      	ffi.copy(i_data, line)
      	i_data = i_data + self.maxFileLength
      	count = count + 1
   	end
   	file:close()
end

function dataset:tableToOutput(dataTable)
   local data
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 3)
   data = torch.Tensor(quantity,sampleSize[1], sampleSize[2], sampleSize[3])
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end
   return data
end

function dataset:sampleHook(path)
   	local input = loadImage(path)
   	local out
   	out = input:clone()
   	return out
end

function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]),self.pathLength[indices[i]])
      local out = self:sampleHook(imgpath)
      table.insert(dataTable,out)
   end
   local data = self:tableToOutput(dataTable)
   return data
end



-- N.LEE
function dataset:get_randomly(nTotal, nSel)
	-- generate randome sequence (like randperm)
	local randSeq = torch.randperm(nTotal):narrow(1,1,nSel) 
	local dataTable = {}
	for i=1, randSeq:size(1) do
		local imgpath = ffi.string(torch.data(self.imagePath[randSeq[i]]), self.pathLength[randSeq[i]])
		local out = self:sampleHook(imgpath)
		table.insert(dataTable, out)
	end
	local data = self:tableToOutput(dataTable)
	return data
end

function dataset:get_randomly_indices(indices)
	local dataTable = {}
	for i=1, indices:size(1) do
		local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.pathLength[indices[i]])
		local out = self:sampleHook(imgpath)
		table.insert(dataTable, out)
	end
	local data = self:tableToOutput(dataTable)

	return data
end

function dataset:get_samplename(idx)
	local imgpath = ffi.string(torch.data(self.imagePath[idx]), self.pathLength[idx])
	return imgpath
end

function dataset:get_substring(imgpath, str, length)
	local _, startIdx = imgpath:find(str)
	startIdx = startIdx + 1
	local foundstr = imgpath:sub(startIdx, startIdx+length)
	return foundstr
end

function dataset:get_jointpath(imgpath)
	local loc 	= self:get_substring(imgpath, 'loc', 5)
	local id 	= self:get_substring(imgpath, 'id', 4)
	local pose  = self:get_substring(imgpath, 'pose', 4)
	local rot   = self:get_substring(imgpath, 'rot', 2)
	local jointpath = '/home/namhoon/develop/PoseRegression/data/rendout/joints_norm/2D/loc' .. loc .. '/Ped_id' .. id .. '_pose' .. pose .. '_rot' .. rot .. '.txt'
	return jointpath
end

function dataset:get_label_fullbody(indices)

	-- for 14 joints model, full body
	local label = torch.Tensor(indices:size(1), 28)  

	for i=1, indices:size(1) do
		local jointpath = self:get_jointpath(self:get_samplename(indices[i]))
		local file = assert(io.open(jointpath, 'r'))
		local cnt = 0
		for line in file:lines() do
			local tmp1 = line:find(' ')
			local x = tonumber(line:sub(1,tmp1-1))
			local y = tonumber(line:sub(tmp1,line:len()))
			cnt = cnt + 1
			label[i][cnt] = x 
			cnt = cnt + 1
			label[i][cnt] = y 
		end
		file:close()
	end

	return label
end

function dataset:get_label_upperbody(indices)

	-- for 14 joints model, only for upper body
	local label = torch.Tensor(indices:size(1), 16)  

	for i=1, indices:size(1) do
		local jointpath = self:get_jointpath(self:get_samplename(indices[i]))
		local file = assert(io.open(jointpath), 'r')

		local cnt = 0
		local cntLine = 0
		for line in file:lines() do
			cntLine = cntLine + 1
			if ((cntLine <= 5) or (cntLine >= 9 and cntLine <=11)) then 
				local tmp1 = line:find(' ')
				local x = tonumber(line:sub(1,tmp1-1))
				local y = tonumber(line:sub(tmp1,line:len()))
				cnt = cnt + 1
				label[i][cnt] = x 
				cnt = cnt + 1
				label[i][cnt] = y 
			end
		end
		file:close()
	end

	return label
end

function dataset:get_label_lowerbody(indices)

	-- for 14 joints model, only for upper body (nJoints=6)
	local label = torch.Tensor(indices:size(1), 12)  

	for i=1, indices:size(1) do
		local jointpath = self:get_jointpath(self:get_samplename(indices[i]))
		local file = assert(io.open(jointpath), 'r')

		local cnt = 0
		local cntLine = 0
		for line in file:lines() do
			cntLine = cntLine + 1
			if ((cntLine >= 12) or (cntLine >= 6 and cntLine <=8)) then 
				local tmp1 = line:find(' ')
				local x = tonumber(line:sub(1,tmp1-1))
				local y = tonumber(line:sub(tmp1,line:len()))
				cnt = cnt + 1
				label[i][cnt] = x 
				cnt = cnt + 1
				label[i][cnt] = y 
			end
		end
		file:close()
	end

	return label
end

function dataset:get_label_fortest(indices, pathtojoints)
	local label = torch.Tensor(indices:size(1),28)

	for i=1, indices:size(1) do
		local jointpath = paths.concat(pathtojoints, string.format('joints2d_%dth.txt',i))
		local file = assert(io.open(jointpath, 'r'))
		local cnt = 0
		for line in file:lines() do
			local tmp1 = line:find(' ')
			local x = tonumber(line:sub(1,tmp1-1))
			local y = tonumber(line:sub(tmp1,line:len()))
			cnt = cnt + 1
			label[i][cnt] = x 
			cnt = cnt + 1
			label[i][cnt] = y 
		end
		file:close()
	end

	return label
end

function dataset:get_label(part, indices, ltTable) 
	local label
	if part == 'fullbody' then
		label =  self:get_label_fullbody(indices)
	elseif part == 'upperbody' then
		label =  self:get_label_upperbody(indices)
	elseif part == 'lowerbody' then
		label =  self:get_label_lowerbody(indices)
	end

	return label
end

function dataset:get_label_filt(part, indices)
	assert(part == 'fullbody')	-- currently only fullbody allowed

	-- load regular labels, which is 28 values for 14 joints
	local label_ori = self:get_label(part, indices)

	-- convert to spatial labels
	local label_filt = convert_labels_to_spatialLabels(label_ori)

	print(label_filt)
	adf=adf+1
	return label_filt, label_ori
end

local function loadImageCrop(path, labelOri)

	-- bw_outer, bh_outer
	local bw_outer = 90
	local bh_outer = bw_outer*2
	local bw_crop = sampleSize[3]
	local bh_crop = sampleSize[2]
	local marginRatio = 0.2

	-- compute bw_tight, bh_tight
	local label_res = torch.reshape(labelOri, 14, 2)
	local label_x_min = torch.min(label_res[{{}, {1}}])
	local label_x_max = torch.max(label_res[{{}, {1}}])
	local label_y_min = torch.min(label_res[{{}, {2}}])
	local label_y_max = torch.max(label_res[{{}, {2}}])

	-- lt_tight, rb_tight
	local lt_tight_x = label_x_min * bw_outer
	local lt_tight_y = label_y_min * bh_outer
	local rb_tight_x = label_x_max * bw_outer
	local rb_tight_y = label_y_max * bh_outer
	local bw_tight = rb_tight_x - lt_tight_x
	local bh_tight = rb_tight_y - lt_tight_y

	-- lt_crop, rb_crop (min and max)
	local lt_crop_x_max, lt_crop_y_max
	if lt_tight_x > bw_outer-bw_crop then 
		lt_crop_x_max = bw_outer - bw_crop
	else 
		lt_crop_x_max = lt_tight_x 
	end
	if lt_tight_y > bh_outer-bh_crop then 
		lt_crop_y_max = bh_outer - bh_crop 
	else 
		lt_crop_y_max = lt_tight_y 
	end
	
	local lt_crop_x_min, lt_crop_y_min
	if rb_tight_x > bw_crop then
		lt_crop_x_min = rb_tight_x - bw_crop
	else
		lt_crop_x_min = 1e-2
	end
	if rb_tight_y > bh_crop then
		lt_crop_y_min = rb_tight_y - bh_crop 
	else
		lt_crop_y_min = 1e-2
	end

	-- randomly select lt
	local lt_x_sel = math.ceil(torch.uniform(lt_crop_x_min, lt_crop_x_max-1))
	local lt_y_sel = math.ceil(torch.uniform(lt_crop_y_min, lt_crop_y_max-1))

	-- load image, resize and crop
	local input = image.load(path, 3, 'float')
	input = image.scale(input, bw_outer .. 'x' .. bh_outer)	
	input = image.crop(input, lt_x_sel, lt_y_sel, lt_x_sel+bw_crop, lt_y_sel+bh_crop)
	assert(input:size(2) == bh_crop and input:size(3) == bw_crop)

	-- transform label 
	local scalar = torch.repeatTensor(torch.Tensor({bw_outer,bh_outer}), 14, 1)
	local label = torch.cmul(label_res, scalar)
	local translation = torch.repeatTensor(torch.Tensor({lt_x_sel,lt_y_sel}),14)
	label:add(-translation)
	scalar = torch.repeatTensor(torch.Tensor({bw_crop,bh_crop}), 14, 1)
	label:cdiv(scalar)
	label = torch.reshape(label, 28)

	-- sanity check
	for i=1, label:size(1) do
		if label[i] <= 0.0 or label[i] >= 1.0 then
			assert(false, 'this should not happen!')
		end
	end

	return input, label
end

function dataset:get_crop_label(indices)
	local imagetensor = torch.Tensor(indices:size(1), 3, 128, 64)
	local labeltensor = torch.Tensor(indices:size(1), 28)

	local labelOri= self:get_label_fullbody(indices)

	for i=1, indices:size(1) do
		local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.pathLength[indices[i]])
		local image, label = loadImageCrop(imgpath, labelOri[i])
		imagetensor[i] = image
		labeltensor[i] = label
	end

	local out = {data = imagetensor, label = labeltensor}

	return out
end

--function dataset:get_label_filt_struct(part, indices)
--	assert(part == 'fullbody')  -- consider all body, but each joints separately
--
--	-- load regular labels, which is 28 values for 14 joints
--	local label_ori = self:get_label(part, indices)
--
--	-- convert to spatial labels
--	local label_filt = convert_labels_to_spatialLabels(label_ori)
--	
--	-- reshape 2688 -> 14*(64+128) : 14 joints separately
--	label_struct = torch.reshape(label_filt, label_filt:size(1), nJoints, W+H)
--	print(label_filt:size())
--	print(label_struct:size())

--	return label_struct

--end


return dataset








