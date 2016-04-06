--[[ 
-- sposenet.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- The base code is inspired by ResNet.
--]]

local nn = require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'


--local Convolution = cudnn.SpatialConvolution
--local Avg = cudnn.SpatialAveragePooling
--local ReLU = cudnn.ReLU
--local Max = nn.SpatialMaxPooling
--local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)

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
        local act1  = cudnn.ReLU(true)(bn1)
        local conv1 = cudnn.SpatialConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act1)
        local bn2   = nn.SpatialBatchNormalization(nChOut)(conv1)
        local act2  = cudnn.ReLU(true)(bn2)
        local conv2 = cudnn.SpatialConvolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act2)

        -- identity shortcut
        local identity = shortcut(nChIn, nChOut, input)

        -- final output
        local output = nn.CAddTable()({identity, conv2})
        return output
    end

    local function CFBlock(nChIn, nChOut, nChConf, input)
        
        -- feature
        local bn1   = nn.SpatialBatchNormalization(nChIn)(input)
        local act1  = cudnn.ReLU(true)(bn1)
        local conv1 = cudnn.SpatialConvolution(nChIn,nChOut-nChConf,1,1,1,1,0,0)(act1)

        -- confidence
        local bn_conf   = nn.SpatialBatchNormalization(nChIn)(input)
        local act_conf  = cudnn.ReLU(true)(bn_conf)
        local conv_conf = cudnn.SpatialConvolution(nChIn,nChConf,1,1,1,1,0,0)(act_conf)

        -- feature + confidence
        local fc = nn.JoinTable(2)({conv1, conv_conf})

        return fc, conv_conf
    end


    local model

    if opt.dataset == 'towncenter' then
        print(' | sposeNet .. towncenter')
        -- sposeNet Towncenter model
       
        local input = nn.Identity()()
        local conv1 = cudnn.SpatialConvolution(3,64,5,5,1,1,2,2)(input)
        local bn1   = nn.SpatialBatchNormalization(64)(conv1)
        local act1  = cudnn.ReLU(true)(bn1)

        local res1  = ResBlock(64, 64, 3, act1)
        local res2  = ResBlock(64, 64, 3, res1)
        local res3  = ResBlock(64, 128, 3, res2)
        local res4  = ResBlock(128, 128, 3, res3)
        local res5  = ResBlock(128, 256, 3, res4)
        local res6  = ResBlock(256, 256, 3, res5)

        -- CPM's second stage
        local cf1, c1  = CFBlock(256,256,30,res6)

        local bn_end1   = nn.SpatialBatchNormalization(256)(cf1)
        local act_end1  = cudnn.ReLU(true)(bn_end1)
        local conv_end1 = cudnn.SpatialConvolution(256,256,9,9,1,1,4,4)(act_end1)
        local bn_end2   = nn.SpatialBatchNormalization(256)(conv_end1)
        local act_end2  = cudnn.ReLU(true)(bn_end2)
        local conv_end2 = cudnn.SpatialConvolution(256,256,11,11,1,1,5,5)(act_end2)
        local bn_end3   = nn.SpatialBatchNormalization(256)(conv_end2)
        local act_end3  = cudnn.ReLU(true)(bn_end3)
        local conv_end3 = cudnn.SpatialConvolution(256,30,1,1)(act_end3)

        --local sum = nn.CAddTable()({c1,c2,c3, conv_end})
        --model = nn.gModule({input}, {c1,c2,c3, sum})

        -- another model. no sum of confidence maps at the end
        -- only have multi outputs to have multiple losses
        model = nn.gModule({input}, {c1, conv_end3})


        -- draw and save model
        graph.dot(model.fg, 'forward graph', './tmp/fg_CPM')
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
       print(1)
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
