--[[
-- eval_jsc.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- evaluate jsc (j27, seg, cen)
--]]

local matio = require 'matio'
--paths.dofile('datanew.lua')

function find_peak(hmap)
    -- returns {ndata x 27 x 2} 
    assert(hmap:size(2) == opt.nJoints and hmap:size(3) == opt.H_jsc and hmap:size(4) == opt.W_jsc)

    local nData = hmap:size(1)
    local jnt = torch.CudaTensor(nData, opt.nJoints, 2)
    local occ = torch.CudaTensor(nData, opt.nJoints, 1)

    for i=1,nData do
        for j=1,opt.nJoints do
            -- find max and idx
            local max, idx = torch.max(torch.reshape(hmap[i][j], opt.H_jsc*opt.W_jsc), 1)
            local k = math.floor(idx[1]/opt.W_jsc)+1
            local l = idx[1] % opt.W_jsc
            -- new label
            jnt[{ {i}, {j}, {1}}] = k
            jnt[{ {i}, {j}, {2}}] = l
            -- save occ. indicator
            if max[1] < 0.9 then   -- This is occluded. Only gt label matters..
                occ[{ {i}, {j}, {1} }] = 1
            else
                occ[{ {i}, {j}, {1} }] = 0
            end
        end
    end

    return jnt, occ
end

function comp_PCK(gt, pred, occ, normscalor)
    -- input: {nData, 27, 2} 
    assert(gt:size(1) == pred:size(1))
    assert(gt:size(2) == opt.nJoints and gt:size(3) == 2)
    
    local nData = gt:size(1)

    local pck_cnt = 0
    local num_occ = 0

    for i=1,nData do 
        local p = pred[i]
        local g = gt[i]

        -- compute the size of head
        local hsize = math.sqrt(math.pow(g[1][1]-g[2][1], 2)+math.pow(g[1][2]-g[2][2], 2))

        -- if distance is lses than normscalor * head size then passed!
        for j=1,opt.nJoints do
            if occ[i][j] == 1 then
                num_occ = num_occ + 1
            else
                local d=torch.sqrt(math.pow(g[j][1]-p[j][1], 2)+math.pow(g[j][2]-p[j][2], 2))
                if d <= hsize * normscalor then
                    pck_cnt = pck_cnt + 1
                end
            end
        end
    end

    local PCK = ( pck_cnt / (opt.nJoints * nData - num_occ) ) * 100
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
local sad_cen -- SAD for cen

local timer = torch.Timer()

function eval_jsc ()
    print('--EVAL_JSDC STARTS..')

    cutorch.synchronize()
    timer:reset()

    -- set the dropouts to evaluate mode
    model:evaluate()
    pck_j27 = 0
    sad_seg = 0
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
    sad_cen = sad_cen / (opt.nTestData/opt.batchSize)

    evalLogger:add{ 
        ['pck_j27'] = pck_j27,
        ['sad_seg'] = sad_seg,
        ['sad_cen'] = sad_cen,
    }

    print(string.format('  PCK (j27):  %.2f ', pck_j27))
    print(string.format('  SAD (seg/cen):  %.2f | %.2f ',sad_seg,sad_cen))
    print(string.format('  time(s): %.2f', timer:time().real))

    collectgarbage()

end

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local outputs
local gt_j27_hmap
local gt_seg
local gt_cen
local pred_j27_hmap
local pred_seg
local pred_cen

local gt_j27
local pred_j27

function evalBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    collectgarbage()
    
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    outputs = model:forward(inputs)
    if torch.type(outputs) == 'table' then
        outputs = outputs[#outputs]
    end

    -- separation
    gt_j27_hmap = labels[{ {}, {1,opt.nJoints}, {}, {} }]
    gt_seg = labels[{ {}, {28}, {}, {} }]
    gt_cen = labels[{ {}, {29}, {}, {} }]
    pred_j27_hmap = outputs[{ {}, {1,opt.nJoints}, {}, {} }]
    pred_seg = outputs[{ {}, {28}, {}, {} }]
    pred_cen = outputs[{ {}, {29}, {}, {} }]

    -- find peak for joints 
    gt_j27   = find_peak(gt_j27_hmap) 
    pred_j27 = find_peak(pred_j27_hmap) 

    -- EVALUATION
    pck_j27 = pck_j27 + comp_PCK(gt_j27, pred_j27)      -- PCK for j27
    sad_seg = sad_seg + comp_SAD(gt_seg, pred_seg)      -- SAD for seg
    sad_cen = sad_cen + comp_SAD(gt_cen, pred_cen)      -- SAD for cen

    cutorch.synchronize()

    collectgarbage()

end







