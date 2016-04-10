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

    local model

    if opt.dataset == 'towncenter' then
        print(' | CPM .. towncenter')
        -- sposeNet Towncenter model
       
        local input = nn.Identity()()

        -- stage1
        local conv1 = cudnn.SpatialConvolution(3,128,5,5,1,1,2,2)(input)
        local act1  = cudnn.ReLU(true)(conv1)
        local pool1 = cudnn.SpatialMaxPooling(3,3,2,2,1,1)(act1) 
        local conv2 = cudnn.SpatialConvolution(128,128,5,5,1,1,2,2)(pool1)
        local act2  = cudnn.ReLU(true)(conv2)
        local pool2 = cudnn.SpatialMaxPooling(3,3,2,2,1,1)(act2) 
        local conv3 = cudnn.SpatialConvolution(128,128,5,5,1,1,2,2)(pool2)
        local act3  = cudnn.ReLU(true)(conv3)
        local conv4 = cudnn.SpatialConvolution(128,32,3,3,1,1,1,1)(act3)
        local act4  = cudnn.ReLU(true)(conv4)
        local conv5 = cudnn.SpatialConvolution(32,512,5,5,1,1,2,2)(act4)
        local act5  = cudnn.ReLU(true)(conv5)
        local conv6 = cudnn.SpatialConvolution(512,512,1,1,1,1,0,0)(act5)
        local act6  = cudnn.ReLU(true)(conv6)
        local conv7 = cudnn.SpatialConvolution(512,28,1,1,1,1,0,0)(act6)

        -- stage2
        local conv1_s2 = cudnn.SpatialConvolution(3,128,5,5,1,1,2,2)(input)
        local act1_s2  = cudnn.ReLU(true)(conv1_s2)
        local pool1_s2 = cudnn.SpatialMaxPooling(3,3,2,2,1,1)(act1_s2) 
        local conv2_s2 = cudnn.SpatialConvolution(128,128,5,5,1,1,2,2)(pool1_s2)
        local act2_s2  = cudnn.ReLU(true)(conv2_s2)
        local pool2_s2 = cudnn.SpatialMaxPooling(3,3,2,2,1,1)(act2_s2) 
        local conv3_s2 = cudnn.SpatialConvolution(128,128,5,5,1,1,2,2)(pool2_s2)
        local act3_s2  = cudnn.ReLU(true)(conv3_s2)
        local conv4_s2 = cudnn.SpatialConvolution(128,32,3,3,1,1,1,1)(act3_s2)
        local act4_s2  = cudnn.ReLU(true)(conv4_s2)

        -- concatenation
        local cat = nn.JoinTable(2)({conv7, act4_s2})

        local conv1_cat = cudnn.SpatialConvolution(60,128,7,7,1,1,3,3)(cat)
        local act1_cat  = cudnn.ReLU(true)(conv1_cat)
        local conv2_cat = cudnn.SpatialConvolution(128,128,7,7,1,1,3,3)(act1_cat)
        local act2_cat  = cudnn.ReLU(true)(conv2_cat)
        local conv3_cat = cudnn.SpatialConvolution(128,128,7,7,1,1,3,3)(act2_cat)
        local act3_cat  = cudnn.ReLU(true)(conv3_cat)
        local conv4_cat = cudnn.SpatialConvolution(128,128,1,1,1,1,0,0)(act3_cat)
        local act4_cat  = cudnn.ReLU(true)(conv4_cat)
        local conv5_cat = cudnn.SpatialConvolution(128,28,1,1,1,1,0,0)(act4_cat)

        
        model = nn.gModule({input}, {conv7, conv5_cat})


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
