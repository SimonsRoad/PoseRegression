--[[ 
-- sposenet.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- The base code is inspired by ResNet.
--]]

local nn = require 'nn'
require 'cunn'
require 'nngraph'


--local Convolution = cudnn.SpatialConvolution
--local Avg = cudnn.SpatialAveragePooling
--local ReLU = cudnn.ReLU
--local Max = nn.SpatialMaxPooling
--local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
    local iChannels

    local function shortcut_basic(nChIn, nChOut)
        if nChIn ~= nChOut then
            return nn.Sequential()
                :add(nn.Concat(2)
                    :add(nn.Identity())
                    :add(nn.MulConstant(0)))
        else
            return nn.Identity()
        end
    end

    local function shortcut_cf(nChIn, nChConf)
        local zeropad = nn.Sequential()
        zeropad:add(Convolution(nChIn,nChConf,1,1))
        zeropad:add(nn.MulConstant(0))
        return nn.Sequential()
            :add(nn.Concat(2)
                :add(nn.Identity())
                :add(zeropad))
    end

    local function resBlock(nChIn, nChOut, sz)

        local s = nn.Sequential()
        s:add(nn.SpatialBatchNormalization(nChIn))
        s:add(nn.ReLU(true))
        s:add(nn.SpatialConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))
        s:add(nn.SpatialBatchNormalization(nChOut))
        s:add(nn.ReLU(true))
        s:add(nn.SpatialConvolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(shortcut_basic(nChIn, nChOut)))
            :add(nn.CAddTable(true))
    end

    local function shortcut(nChIn, nChOut, input)
        if nChIn ~= nChOut then
            local identity = nn.Identity()(input)
            local padding  = nn.MulConstant(0)(input)
            return nn.JoinTable(2)({identity, padding})
        else
            return nn.Identity()(input) 
        end
    end

    local function ResBlock(nChIn, nChOut, sz, input)
        
        -- feature
        local bn1   = nn.SpatialBatchNormalization(nChIn)(input)
        local act1  = nn.ReLU(true)(bn1)
        local conv1 = nn.SpatialConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act1)
        local bn2   = nn.SpatialBatchNormalization(nChOut)(conv1)
        local act2  = nn.ReLU(true)(bn2)
        local conv2 = nn.SpatialConvolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act2)

        -- identity shortcut
        local identity = shortcut(nChIn, nChOut, input)

        -- final output
        local output = nn.CAddTable()({identity, conv2})
        return output
    end

    local function CFBlock(nChIn, nChOut, nChConf, sz, input)
        
        -- feature
        local bn1   = nn.SpatialBatchNormalization(nChIn)(input)
        local act1  = nn.ReLU(true)(bn1)
        local conv1 = nn.SpatialConvolution(nChIn,nChOut-nChConf,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act1)
        local bn2   = nn.SpatialBatchNormalization(nChOut-nChConf)(conv1)
        local act2  = nn.ReLU(true)(bn2)
        local conv2 = nn.SpatialConvolution(nChOut-nChConf,nChOut-nChConf,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act2)

        -- confidence
        local bn_conf   = nn.SpatialBatchNormalization(nChIn)(input)
        local act_conf  = nn.ReLU(true)(bn_conf)
        local conv_conf = nn.SpatialConvolution(nChIn,nChConf,1,1,1,1,0,0)(act_conf)

        -- feature + confidence
        local fc = nn.JoinTable(2)({conv2, conv_conf})

        -- identity shortcut
        local identity = shortcut(nChIn, nChOut, input)

        -- final output
        local output = nn.CAddTable()({identity, fc})

        return output, conv_conf
    end


    local model

    if opt.dataset == 'towncenter' then
        print(' | sposeNet .. towncenter')
        -- sposeNet Towncenter model
       
        local input = nn.Identity()()
        local conv1 = nn.SpatialConvolution(3,64,5,5,1,1,2,2)(input)
        local bn1   = nn.SpatialBatchNormalization(64)(conv1)
        local act1  = nn.ReLU(true)(bn1)

        local res1  = ResBlock(64, 64, 3, act1)
        local res2  = ResBlock(64, 128, 3, res1)
        local res3  = ResBlock(128, 256, 3, res2)
        local res4  = ResBlock(256, 256, 3, res3)

        local cf1, c1  = CFBlock(256,256,30,9,res4)
        local cf2, c2  = CFBlock(256,256,30,11,cf1)
        local cf3, c3  = CFBlock(256,256,30,13,cf2)

        local bn_end   = nn.SpatialBatchNormalization(256)(cf3)
        local act_end  = nn.ReLU(true)(bn_end)
        local conv_end = nn.SpatialConvolution(256,30,1,1)(act_end)

        local sum = nn.CAddTable()({c1,c2,c3, conv_end})

        model = nn.gModule({input}, {c1, c2, c3, sum})

        -- draw and save model
        graph.dot(model.fg, 'forward graph', './tmp/fg')
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   --[[
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
   --]]
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
