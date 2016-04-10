--[[
-- eval_jsdc.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- evaluate jsdc (j27, seg, dep, cen)
--]]

local matio = require 'matio'

function find_peak(hmap)
    -- 1. hmap expects {ndata x 27 x 128 x 64}
    -- 2. returns {ndata x 27 x 2} 
    assert(hmap:size(2) == opt.nJoints and hmap:size(3) == opt.H_jsdc and hmap:size(4) == opt.W_jsdc)

    local nData = hmap:size(1)
    local j27 = torch.CudaTensor(nData, opt.nJoints, 2)

    for i=1,nData do
        for j=1,opt.nJoints do
            -- find max and idx
            local max, idx = torch.max(torch.reshape(hmap[i][j], opt.H_jsdc*opt.W_jsdc), 1)
            local k = math.floor(idx[1]/opt.W_jsdc)+1
            local l = idx[1] % opt.W_jsdc
            -- new label
            j27[{ {i}, {j}, {1}}] = k
            j27[{ {i}, {j}, {2}}] = l
        end
    end

    return j27
end

function comp_PCK(gt, pred)
    -- input: {nData, 27, 2} 
    assert(gt:size(1) == pred:size(1))
    assert(gt:size(2) == opt.nJoints and gt:size(3) == 2)
    
    local nData = gt:size(1)

    local alpha = 0.5
    local pck_cnt = 0

    for i=1,nData do 
        local p = pred[i]
        local g = gt[i]

        -- compute the size of head
        local hsize = math.sqrt(math.pow(g[1][1]-g[2][1], 2)+math.pow(g[1][2]-g[2][2], 2))

        -- if distance is lses than alpha * head size then passed!
        for j=1,opt.nJoints do
            local d=torch.sqrt(math.pow(g[j][1]-p[j][1], 2)+math.pow(g[j][2]-p[j][2], 2))
            if d <= hsize * alpha then
                pck_cnt = pck_cnt + 1
            end
        end
    end

    local PCK = ( pck_cnt / (opt.nJoints * nData) ) * 100
    return PCK
end


function comp_SAD (gt, pred) -- average squard distance
    assert(gt:size(1) == pred:size(1))
    assert(gt:nDimension() == pred:nDimension())
    assert(gt:nDimension() == 4)

    local sad
    sad = torch.sum(torch.abs(gt-pred)) / gt:size(1) 
    return sad
end

evalLogger = optim.Logger(paths.concat(opt.save, 'eval.log'))

local pck_j27 -- PCK for j27
local sad_seg -- SAD for seg
local sad_dep -- SAD for dep
local sad_cen -- SAD for cen

local timer = torch.Timer()

function eval_jsdc ()
    print('--EVAL_JSDC STARTS..')

    cutorch.synchronize()
    timer:reset()

    -- set the dropouts to evaluate mode
    model:evaluate()
    pck_j27 = 0
    sad_seg = 0
    sad_dep = 0
    sad_cen = 0

    -- randomize dataset (actually just indices)
    --local idx_rand = torch.randperm(opt.nTestData)
    local idx_rand = torch.range(1, opt.nTestData)

    for i=1,opt.nTestData/opt.batchSize do
        local idx_start = (i-1) * opt.batchSize + 1
        local idx_end   = idx_start + opt.batchSize - 1
        local idx_batch
        if idx_end <= opt.nTestData then
            idx_batch = idx_rand[{{idx_start,idx_end}}]
        else
            local idx1 = idx_rand[{{idx_start,opt.nTestData}}]
            local idx2 = idx_rand[{{1,idx_end-opt.nTestData}}]
            idx_batch = torch.cat(idx1, idx2, 1)
        end
        idx_batch = idx_batch + opt.nTrainData

        donkeys:addjob(
            function()
                return loader:load_batch_new(idx_batch)
            end,
            evalBatch
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()

    pck_j27 = pck_j27 / (opt.nTestData/opt.batchSize)
    sad_seg = sad_seg / (opt.nTestData/opt.batchSize)
    sad_dep = sad_dep / (opt.nTestData/opt.batchSize)
    sad_cen = sad_cen / (opt.nTestData/opt.batchSize)

    -- test on Real images 
    pck_j14_real = testOnReal()

    evalLogger:add{ 
        ['pck_j27'] = pck_j27,
        ['sad_seg'] = sad_seg,
        ['sad_dep'] = sad_dep,
        ['sad_cen'] = sad_cen,
    }

    print(string.format('  PCK (j27):  %.2f ', pck_j27))
    print(string.format('  SAD (seg/dep/cen):  %.2f | %.2f | %.2f ',sad_seg,sad_dep,sad_cen))
    print(string.format('  time(s): %.2f', timer:time().real))

    collectgarbage()

end

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local outputs
local gt_j27_hmap
local gt_seg
local gt_dep
local gt_cen
local pred_j27_hmap
local pred_seg
local pred_dep
local pred_cen

local gt_j27
local pred_j27

function evalBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    collectgarbage()
    
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    outputs = model:forward(inputs)
    assert(torch.type(outputs)=='table') -- multiple outputs as table
    outputs = outputs[#outputs]


    -- separation
    gt_j27_hmap = labels[{ {}, {1,opt.nJoints}, {}, {} }]
    gt_seg = labels[{ {}, {28}, {}, {} }]
    gt_dep = labels[{ {}, {29}, {}, {} }]
    gt_cen = labels[{ {}, {30}, {}, {} }]
    pred_j27_hmap = outputs[{ {}, {1,opt.nJoints}, {}, {} }]
    pred_seg = outputs[{ {}, {28}, {}, {} }]
    pred_dep = outputs[{ {}, {29}, {}, {} }]
    pred_cen = outputs[{ {}, {30}, {}, {} }]
    print(pred_cen)
    adf=adf+1

    -- find peak for joints 
    gt_j27   = find_peak(gt_j27_hmap) 
    pred_j27 = find_peak(pred_j27_hmap) 

    -- EVALUATION
    pck_j27 = pck_j27 + comp_PCK(gt_j27, pred_j27)      -- PCK for j27
    sad_seg = sad_seg + comp_SAD(gt_seg, pred_seg)      -- SAD for seg
    sad_dep = sad_dep + comp_SAD(gt_dep, pred_dep)      -- SAD for dep
    sad_cen = sad_cen + comp_SAD(gt_cen, pred_cen)      -- SAD for cen

    cutorch.synchronize()

    collectgarbage()

end

function testOnReal()

    -- load rTest
    local loaderReal = dataLoader{txtimg=opt.txtimgreal, txtjsdc=opt.txtjsdcreal}

    -- compute PCK
    local numReal = loaderReal:size()
    assert(numReal == 22)
    local pcksum = 0
    for i=1,numReal do
        local testindices = torch.Tensor({i})
        local testimg  = loaderReal:load_img(testindices)
        local testjsdc = loaderReal:load_jsdc(testindices)
        for j=1,3 do
            testimg[{ {}, {j}, {}, {} }]:add(-mean[j])
            testimg[{ {}, {j}, {}, {} }]:div(std[j])
        end
        local output = model:forward(testimg:cuda())
        if torch.type(output) == 'table' then
            output = output[#output]
        end

        -- compute PCK
        local gt_j14_hmap   = testjsdc[{ {}, {1,14}, {}, {} }]
        local pred_j14_hmap = output[{ {}, {1,14}, {}, {} }] 
        local gt_j14   = find_peak(gt_j14_hmap)
        local pred_j14 = find_peak(pred_j14_hmap)
        local pck = comp_PCK(gt_j14, pred_j14)
        print(string.format('PCK [real] (image %d): %.2f', i, pck))
        pcksum = pcksum + pck
    end
    print(string.format('PCK [real] total: %.2f', pcksum/numReal))

    return pcksum/numReal

end







