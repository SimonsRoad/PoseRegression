--[[
-- datanew.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]

torch.setdefaulttensortype('torch.FloatTensor')
local ffi       = require 'ffi'
local class     = require('pl.class')
local dir       = require 'pl.dir'
local tablex    = require 'pl.tablex'
local argcheck  = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
local matio = require 'matio'


local imgtensor = torch.Tensor(opt.batchSize, 3, opt.H, opt.W) 
local jsdc_tensor = torch.Tensor(opt.batchSize, opt.nChOut, opt.H_jsdc, opt.W_jsdc) 

local dataset = torch.class('dataLoader')
local initcheck = argcheck{
   pack=true,
    {name="txtimg",
    type="string",
    help=""},
    {name="txtjsdc",
    type="string",
    help=""}
}

function dataset:__init(...)
	local args =  initcheck(...)
	for k,v in pairs(args) do self[k] = v end
	self.imageNumSamples = tonumber(sys.fexecute("cat " .. self.txtimg .. " |  wc -l"))
	self.imageMaxFileLength = tonumber(sys.fexecute("cat " .. self.txtimg .. " |  awk '{print length($0)}' | datamash max 1"))
   	self.imagePath = torch.CharTensor() -- path to each image in dataset
	self.imagePath:resize(self.imageNumSamples, self.imageMaxFileLength)
   	local i_data = self.imagePath:data()
   	local file = assert(io.open(self.txtimg, "r"))
   	self.imagePathLength = torch.LongTensor(self.imageNumSamples):fill(0)   
   	local count = 1
   	for line in file:lines() do
    	self.imagePathLength[count] = line:len()
      	ffi.copy(i_data, line)
      	i_data = i_data + self.imageMaxFileLength
      	count = count + 1
   	end
   	file:close()

	self.labelNumSamples = tonumber(sys.fexecute("cat " .. self.txtjsdc.. " |  wc -l"))
    assert(self.imageNumSamples == self.labelNumSamples)
	self.labelMaxFileLength = tonumber(sys.fexecute("cat " .. self.txtjsdc.. " |  awk '{print length($0)}' | datamash max 1"))
   	self.labelPath = torch.CharTensor() -- path to each label in dataset
	self.labelPath:resize(self.labelNumSamples, self.labelMaxFileLength)
   	local i_data = self.labelPath:data()
   	local file = assert(io.open(self.txtjsdc, "r"))
   	self.labelPathLength = torch.LongTensor(self.labelNumSamples):fill(0)   
   	local count = 1
   	for line in file:lines() do
    	self.labelPathLength[count] = line:len()
      	ffi.copy(i_data, line)
      	i_data = i_data + self.labelMaxFileLength
      	count = count + 1
   	end
   	file:close()
    self.numSamples = self.imageNumSamples
end

function dataset:size()
    return self.numSamples
end

function dataset:load_img(indices)
    -- images 
    local img = torch.Tensor(indices:size(1), 3, opt.H, opt.W)
    for i=1, indices:size(1) do
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.imagePathLength[indices[i]])
        img[i] = image.load(imgpath, 3, 'float')
    end
    return img
end

function dataset:load_jsdc(indices)
    -- load from .mat files 
    local jsdc = torch.Tensor(indices:size(1), opt.nChOut, opt.H_jsdc, opt.W_jsdc)
    for i=1, indices:size(1) do
        local jsdc_path = ffi.string(torch.data(self.labelPath[indices[i]]), self.labelPathLength[indices[i]])
        --jsdc[i] = matio.load(jsdc_path, 'jsdc')
        jsdc[i] = matio.load(jsdc_path, 'hmap_downsize')
    end
    return jsdc
end


--[[
function dataset:load_batch(indices)
    -- load all
    pos[{}] = self:load_img(indices)
    jsdc[{}] = self:load_jsdc(indices)
        
    -- normalize images (pos)
    for i=1,3 do
        pos[{ {}, {i}, {}, {} }]:add(-mean[i])
        pos[{ {}, {i}, {}, {} }]:div(std[i])
    end

    -- out
    local out = {data = pos, label = jsdc}
    return out
end
--]]

function dataset:load_batch_new(indices)
    -- images
    for i=1, indices:size(1) do
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.imagePathLength[indices[i]])
        imgtensor[i] = image.load(imgpath, 3, 'float')
    end
    -- load from .mat files 
    for i=1, indices:size(1) do
        local jsdc_path = ffi.string(torch.data(self.labelPath[indices[i]]), self.labelPathLength[indices[i]])
        --jsdc_tensor[i] = matio.load(jsdc_path, 'jsdc')
        jsdc_tensor[i] = matio.load(jsdc_path, 'hmap_downsize')
    end

    -- normalize images (pos)
    for i=1,3 do
        imgtensor[{ {}, {i}, {}, {} }]:add(-mean[i])
        imgtensor[{ {}, {i}, {}, {} }]:div(std[i])
    end

    return imgtensor, jsdc_tensor

end


return dataset





