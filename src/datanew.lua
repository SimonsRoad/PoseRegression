-- datanew.lua

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


local dataset = torch.class('dataLoader')
local initcheck = argcheck{
   pack=true,
    {name="txtpos",
    type="string",
    help=""},
    {name="txtjsdc",
    type="string",
    help=""}
}

function dataset:__init(...)
	local args =  initcheck(...)
	for k,v in pairs(args) do self[k] = v end
	self.imageNumSamples = tonumber(sys.fexecute("cat " .. self.txtpos .. " |  wc -l"))
	self.imageMaxFileLength = tonumber(sys.fexecute("cat " .. self.txtpos .. " |  awk '{print length($0)}' | datamash max 1"))
   	self.imagePath = torch.CharTensor() -- path to each image in dataset
	self.imagePath:resize(self.imageNumSamples, self.imageMaxFileLength)
   	local i_data = self.imagePath:data()
   	local file = assert(io.open(self.txtpos, "r"))
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
   	self.labelPath = torch.CharTensor() -- path to each image in dataset
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
	local imgtensor = torch.Tensor(indices:size(1), 3, opt.H, opt.W)
    for i=1, indices:size(1) do
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.imagePathLength[indices[i]])
        local img = image.load(imgpath, 3, 'float')
        imgtensor[i] = img
    end
    
    return imgtensor
end

function dataset:load_jsdc(indices)
    -- load from .mat files 
    local jsdc_tensor = torch.Tensor(indices:size(1), 30, opt.H, opt.W)

    for i=1, indices:size(1) do
        local jsdc_path = ffi.string(torch.data(self.labelPath[indices[i]]), self.labelPathLength[indices[i]])
        local jsdc_table = matio.load(jsdc_path)
        jsdc_tensor[i] = jsdc_table.jsdc
    end

    return jsdc_tensor
end

function dataset:load_batch(indices)
     -- load all
     local pos  = self:load_img(indices)
     local jsdc = self:load_jsdc(indices)
     
    -- normalize images (pos)
    for i=1,3 do
        pos[{ {}, {i}, {}, {} }]:add(-mean[i])
        pos[{ {}, {i}, {}, {} }]:div(std[i])
        
    end

    -- out
    local out = {data = pos, label = jsdc}
    return out
end


return dataset





