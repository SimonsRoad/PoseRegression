--[[ 
-- sposenet.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- The base code is inspired by ResNet.
--]]

require 'nn'
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
        local bn1   = cudnn.SpatialBatchNormalization(nChIn)(input)
        local act1  = cudnn.ReLU(true)(bn1)
        local conv1 = cudnn.SpatialConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act1)
        local bn2   = cudnn.SpatialBatchNormalization(nChOut)(conv1)
        local act2  = cudnn.ReLU(true)(bn2)
        local conv2 = cudnn.SpatialConvolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act2)

        -- identity shortcut
        local identity = shortcut(nChIn, nChOut, input)

        -- final output
        local output = nn.CAddTable()({identity, conv2})
        return output
    end

    local function CFBlock(nChIn, nChOut, nChConf, nChFC, sz, input)
        
        -- feature
        local bn1   = cudnn.SpatialBatchNormalization(nChIn)(input)
        local act1  = cudnn.ReLU(true)(bn1)
        local conv1 = cudnn.SpatialConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act1)
        local bn2   = cudnn.SpatialBatchNormalization(nChOut)(conv1)
        local act2  = cudnn.ReLU(true)(bn2)
        local conv2 = cudnn.SpatialConvolution(nChOut,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)(act2)

        -- confidence
        local bn_conf1   = cudnn.SpatialBatchNormalization(nChIn)(input)
        local act_conf1  = cudnn.ReLU(true)(bn_conf1)
        local conv_conf1 = cudnn.SpatialConvolution(nChIn,nChFC,1,1,1,1,0,0)(act_conf1)
        local bn_conf2   = cudnn.SpatialBatchNormalization(nChFC)(conv_conf1)
        local act_conf2  = cudnn.ReLU(true)(bn_conf2)
        local conv_conf2 = cudnn.SpatialConvolution(nChFC,nChConf,1,1,1,1,0,0)(act_conf2)

        -- identity shortcut
        local identity = shortcut(nChIn, nChOut, input)

        -- identity addition
        local id = nn.CAddTable()({identity, conv2})

        -- feature + confidence
        local output = nn.JoinTable(2)({id, conv_conf2})

        return output, conv_conf2
    end


    local model

    if opt.dataset == 'towncenter' then
        print(' | testmodel2 .. towncenter')
       
        local input = nn.Identity()()
        local conv1 = cudnn.SpatialConvolution(3,64,5,5,1,1,2,2)(input)
        local bn1   = cudnn.SpatialBatchNormalization(64)(conv1)
        local act1  = cudnn.ReLU(true)(bn1)

        local res1  = ResBlock(64, 64, 3, act1)
        local res2  = ResBlock(64, 64, 3, res1)
        local res3  = ResBlock(64, 128, 3, res2)
        local res4  = ResBlock(128, 128, 3, res3)

        local cf1, c1  = CFBlock(128+opt.nChOut*0,128+opt.nChOut*0,opt.nChOut,256,17,res4)
        local cf2, c2  = CFBlock(128+opt.nChOut*1,128+opt.nChOut*1,opt.nChOut,256,17,cf1)
        local cf3, c3  = CFBlock(128+opt.nChOut*2,128+opt.nChOut*2,opt.nChOut,256,17,cf2)

        local bn_end1   = cudnn.SpatialBatchNormalization(128+opt.nChOut*3)(cf3)
        local act_end1  = cudnn.ReLU(true)(bn_end1)
        local conv_end1 = cudnn.SpatialConvolution(128+opt.nChOut*3,256,1,1)(act_end1)
        local bn_end2   = cudnn.SpatialBatchNormalization(256)(conv_end1)
        local act_end2  = cudnn.ReLU(true)(bn_end2)
        local conv_end2 = cudnn.SpatialConvolution(256,opt.nChOut,1,1)(act_end2)

        local cat_last  = nn.JoinTable(2)({c1,c2,c3, conv_end2})

        local bn_cat1   = cudnn.SpatialBatchNormalization(opt.nChOut*4)(cat_last)
        local act_cat1  = cudnn.ReLU(true)(bn_cat1)
        local conv_cat1 = cudnn.SpatialConvolution(opt.nChOut*4,256,1,1)(act_cat1)
        local bn_cat2   = cudnn.SpatialBatchNormalization(256)(conv_cat1)
        local act_cat2  = cudnn.ReLU(true)(bn_cat2)
        local conv_cat2 = cudnn.SpatialConvolution(256,opt.nChOut,1,1)(act_cat2)

        --model = nn.gModule({input}, {conv_end2, conv_cat2})
		-- only one loss!
        model = nn.gModule({input}, {conv_cat2})


        -- draw and save model
        graph.dot(model.fg, 'forward graph', './graphs/testmodel2')
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
   
   --ConvInit('cudnn.SpatialConvolution')
   --ConvInit('nn.SpatialConvolution')
   --BNInit('fbnn.SpatialBatchNormalization')
   --BNInit('cudnn.SpatialBatchNormalization')
   --BNInit('nn.SpatialBatchNormalization')
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
