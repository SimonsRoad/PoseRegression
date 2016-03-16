--[[
-- eval_jsdc.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- evaluate jsdc (j27, seg, dep, cen)
--]]

local matio = require 'matio'

function find_peak(hmap)
    -- 1. hmap expects {ndata x 27 x 128 x 64}
    -- 2. returns {ndata x 27 x 2} 
    assert(hmap:size(2) == 27 and hmap:size(3) == 128 and hmap:size(4) == 64)

    local nData = hmap:size(1)
    local j27 = torch.CudaTensor(nData, 27, 2)

    for i=1,nData do
        for j=1,27 do
            -- find max and idx
            local max, idx = torch.max(torch.reshape(hmap[i][j], 128*64), 1)
            local k = math.floor(idx[1]/64)+1
            local l = idx[1] % 64
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
    assert(gt:size(2) == nJoints and gt:size(3) == 2)
    
    local nData = gt:size(1)

    local alpha = 0.5
    local pck_cnt = 0

    for i=1,nData do 
        local p = pred[i]
        local g = gt[i]

        -- compute the size of head
        local hsize = math.sqrt(math.pow(g[1][1]-g[2][1], 2)+math.pow(g[1][2]-g[2][2], 2))

        -- if distance is lses than alpha * head size then passed!
        for j=1,nJoints do
            local d=math.sqrt(math.pow(g[j][1]-p[j][1], 2)+math.pow(g[j][2]-g[j][2], 2))
            if d <= hsize * alpha then
                pck_cnt = pck_cnt + 1
            end
        end
    end

    local PCK = ( pck_cnt / (nJoints * nData) ) * 100
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

evalLogger = optim.Logger(paths.concat(opt.save, 'pck.log'))

local pck_j27 -- PCK for j27
local sad_seg -- SAD for seg
local sad_dep -- SAD for dep
local sad_cen -- SAD for cen

local timer = torch.Timer()

function eval_jsdc ()

    cutorch.synchronize()
    timer:reset()

    -- set the dropouts to evaluate mode
    model:evaluate()
    pck_j27 = 0
    sad_seg = 0
    sad_dep = 0
    sad_cen = 0

    -- randomize dataset (actually just indices)
    local idx_rand = torch.randperm(opt.nTestData)

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
                local testset_batch = loader:load_batch(idx_batch)
                return testset_batch.data, testset_batch.label
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

    evalLogger:add{ 
        ['pck_j27'] = pck_j27,
        ['sad_seg'] = sad_seg,
        ['sad_dep'] = sad_dep,
        ['sad_cen'] = sad_cen,
    }

    print(string.format('--EVALUATION  [%.2fsec]', timer:time().real))
    print(string.format('  PCK (j27):  %.2f ', pck_j27))
    print(string.format('  SAD (seg/dep/cen):  %.2f | %.2f | %.2f ',sad_seg,sad_dep,sad_cen))

end

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function evalBatch(inputsCPU, labelsCPU)
    
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local outputs = model:forward(inputs)

    local gt = labelsCPU:cuda()
    local pred = outputs

    -- separation
    local gt_j27_hmap = gt[{ {}, {1,27}, {}, {} }]
    local gt_seg = gt[{ {}, {28}, {}, {} }]
    local gt_dep = gt[{ {}, {29}, {}, {} }]
    local gt_cen = gt[{ {}, {30}, {}, {} }]
    local pred_j27_hmap = pred[{ {}, {1,27}, {}, {} }]
    local pred_seg = pred[{ {}, {28}, {}, {} }]
    local pred_dep = pred[{ {}, {29}, {}, {} }]
    local pred_cen = pred[{ {}, {30}, {}, {} }]

    -- find peak for joints 
    local gt_j27   = find_peak(gt_j27_hmap) 
    local pred_j27 = find_peak(pred_j27_hmap) 

    -- EVALUATION
    pck_j27 = pck_j27 + comp_PCK(gt_j27, pred_j27)      -- PCK for j27
    sad_seg = sad_seg + comp_SAD(gt_seg, pred_seg)      -- SAD for seg
    sad_dep = sad_dep + comp_SAD(gt_dep, pred_dep)      -- SAD for dep
    sad_cen = sad_cen + comp_SAD(gt_cen, pred_cen)      -- SAD for cen

    cutorch.synchronize()

end