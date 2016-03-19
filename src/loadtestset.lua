--[[
-- loadtestset.lua
--
--]]

torch.setdefaulttensortype('torch.FloatTensor')
local ffi       = require 'ffi'
local class     = require 'pl.class'
local dir       = require 'pl.dir'
local tablex    = require 'pl.tablex'
local argcheck  = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
local matio = require 'matio'

local testset = torch.class('dataLoader')
local initcheck = argcheck{
    pack=true,
    {name="txttest",
    type="string",
    help=""}
}

function testset:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    self.imageNumSamples = tonumber(sys.fexecute("cat " .. self.txttest .. " | wc -l"))
    self.imageMaxFileLength = tonumber(sys.fexecute("cat " .. self.txttest .. " | awk '{print length($0)}' | datamash max 1"))
    self.imagePath = torch.CharTensor() -- path to each image in dataset
    self.imagePath:resize(self.imageNumSamples, self.imageMaxFileLength)
    local i_data = self.imagePath:data()
    local file = assert(io.open(self.txttest, "r"))
    self.imagePathLength = torch.LongTensor(self.imageNumSamples):fill(0)
    local count = 1
    for line in file:lines() do
        self.imagePathLength[count] = line:len()
        ffi.copy(i_data, line)
        i_data = i_data + self.imageMaxFileLength
        count = count + 1
    end
    file:close()
end

function testset:load_img(indices)

    -- images
    local imgtensor = torch.Tensor(indices:size(1), 3, opt.H, opt.W)
    for i=1, indices:size(1) do
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]), self.imagePathLength[indices[i]])
        local img = image.load(imgpath, 3, 'float')
        img = image.scale(img, '64x128')
        imgtensor[i] = img
    end

    return imgtensor
end












