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
      
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
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

function dataset:get_label(part, indices) 
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
	local label = convert_labels_to_spatialLabels(label_ori)
	return label, label_ori
end


return dataset








