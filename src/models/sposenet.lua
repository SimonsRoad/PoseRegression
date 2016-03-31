--[[ 
-- sposenet.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- The base code is inspired by ResNet.
--]]

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

    local function preActBlock(n, stride)
        local nInputPlane = iChannels
        iChannels = n

        local s = nn.Sequential()
        s:add(SBatchNorm(nInputPlane))
        s:add(ReLU(true))
        s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
        s:add(SBatchNorm(n))
        s:add(ReLU(true))
        s:add(Convolution(n,n,3,3,1,1,1,1))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nInputPlane, n, stride)))
            :add(nn.CAddTable(true))
   end

   local function resBlock(nChIn, nChOut, sz)

       local s = nn.Sequential()
       s:add(SBatchNorm(nChIn))
       s:add(ReLU(true))
       s:add(Convolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))
       s:add(SBatchNorm(nChOut))
       s:add(ReLU(true))
       s:add(Convolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))

       return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut(nChIn, nChOut, 1)))
            :add(nn.CAddTable(true))
   end

    local function shortcut_new(nChIn, nChConf)
        local zeropad = nn.Sequential()
        zeropad:add(Convolution(nChIn,nChConf,1,1))
        zeropad:add(nn.MulConstant(0))
        return nn.Sequential()
            :add(nn.Concat(2)
                :add(nn.Identity())
                :add(zeropad))
   end

   local function confFeatBlock(nChIn, nChOut, nChConf, sz)

       local f = nn.Sequential()
       f:add(SBatchNorm(nChIn))
       f:add(ReLU(true))
       f:add(Convolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))
       f:add(SBatchNorm(nChOut))
       f:add(ReLU(true))
       f:add(Convolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))

       local c = nn.Sequential()
       c:add(SBatchNorm(nChIn))
       c:add(ReLU(true))
       c:add(Convolution(nChIn,nChConf,sz,sz,1,1,(sz-1)/2,(sz-1)/2))

       local fc = nn.Concat(2)
            :add(f)
            :add(c)

       return nn.Sequential()
            :add(nn.ConcatTable()
                :add(fc)
                :add(shortcut_new(nChIn, nChConf)))
            :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()

   if opt.dataset == 'towncenter' then
       print(' | sposeNet .. towncenter')
       -- (Modified) sposeNet Towncenter model
       model:add(Convolution(3,64,5,5,1,1,2,2))
       model:add(SBatchNorm(64))
       model:add(ReLU(true))
       model:add(Max(3,3,1,1,1,1))

       model:add(resBlock(64, 64, 3))
       model:add(resBlock(64, 128, 3))
       model:add(resBlock(128, 128, 3))
       model:add(resBlock(128, 256, 3))
       model:add(resBlock(256, 256, 3))
       model:add(resBlock(256, 512, 3))
       model:add(resBlock(512, 512, 3))

       model:add(confFeatBlock(512, 512, 30, 17))
       model:add(confFeatBlock(512+30, 512+30, 30, 17))

       model:add(SBatchNorm(512+30+30))
       model:add(ReLU(true))
       model:add(Convolution(512+30+30,30,1,1))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
