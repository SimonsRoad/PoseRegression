--[[
--test_pr.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--
-- set opt.nDonkeys 1 (no use of donkey, directly load data)
-- set opt.txtimg, opt.txtjsc for real data. 
-- set pathToModel
-- set mNum
-- test data needs to be pre-processed (resize!)
-- prepare a .txt for test data
--]]

require 'torch'
require 'paths'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'nngraph'

local matio     = require 'matio'
local models    = require 'models/init'
local opts      = require 'opts'
opt = opts.parse(arg)
paths.dofile('datanew.lua')
paths.dofile('eval_jsc.lua')
paths.dofile('load_dataset_config.lua')

-- **SETTING**
opt.nDonkeys = 3


-- load dataset configuration
-- for towncenter [11: generic, 12:uniform, 13:prior x3]
-- for pet2006 [5: 1CF Block, 6: 2CF Block, ]
selLoc = 14                 
datasetname = 'towncenter'
load_dataset_config(datasetname)
opt.W = WH[selLoc][1]
opt.H = WH[selLoc][2]
opt.W_jsc = opt.W
opt.H_jsc = opt.H
opt.nJoints = 14 -- compute PCK is based on only 14 joints


-- TEST DATA: 1) sTrain, 2) sTest, 3) rTest
testsettype = 'sTest'


-- load model
--for mNum = bestmodel[selLoc],bestmodel[selLoc] do
for mNum = 30,30 do

    local mName = string.format('clear_model_%d.t7', mNum)
    local pathToModel = string.format('../save/PR_fcn/option/%s', datetime[selLoc])
    opt.retrain = paths.concat(pathToModel, mName)  
    local model, criterion = models.setup(opt)
    model:cuda()

    savedir = paths.concat(pathToModel, string.format('results/%s/model%d',testsettype,mNum))
    os.execute('mkdir -p ' .. savedir)

    if testsettype == 'sTrain' then
        --opt.txtimg  = string.format('../data/rendout/anc_y%d_x%d/lists/img_pos.txt', y, x)
        --opt.txtjsc  = string.format('../data/rendout/anc_y%d_x%d/lists/jsc_pos.txt', y, x)
    elseif testsettype == 'sTest' then
        opt.txtimg  = string.format('../data/rendout/anc_y%03d_x%d/lists/img_sTest.txt',YX[selLoc][1],YX[selLoc][2])
        opt.txtjsc  = string.format('../data/rendout/anc_y%03d_x%d/lists/jsc_sTest.txt',YX[selLoc][1],YX[selLoc][2])
    elseif testsettype == 'rTest' then
        opt.txtimg  = string.format('../../%s/data/frames_y%03d_x%d_LQ/lists/img_pos.txt', datasetname,YX[selLoc][1],YX[selLoc][2])
        opt.txtjsc  = string.format('../../%s/data/frames_y%03d_x%d_LQ/lists/jsc_pos.txt', datasetname,YX[selLoc][1],YX[selLoc][2])
        --opt.txtimg  = string.format('../../towncenter/data/frames_y%d_x%d_LQ/lists_tmp/img_pos.txt', y, x)
        --opt.txtjsc  = string.format('../../towncenter/data/frames_y%d_x%d_LQ/lists_tmp/jsc_pos.txt', y, x)
    end

    -- load meanstd
    meanstdCache = paths.concat(opt.cache, string.format('meanstdCache/y%d_x%d.t7',YX[selLoc][1],YX[selLoc][2]))
    meanstd = torch.load(meanstdCache)
    mean = meanstd.mean
    std  = meanstd.std

    -- Test
    local loader = dataLoader{txtimg=opt.txtimg, txtjsc=opt.txtjsc}
    local numimages = loader:size()
    print(string.format('number of test images: %d', numimages))
    local numnormscalor = 11
    local pck_all = torch.Tensor(numimages, numnormscalor)
    local indices = torch.Tensor(1):long()
    for i=1,numimages do
        -- load 
        indices[1] = i
        local testimg = loader:load_img(indices)
        local testjsc = loader:load_jsc(indices)
        for j=1,3 do
            testimg[{ {}, {j}, {}, {} }]:add(-mean[j])
            testimg[{ {}, {j}, {}, {} }]:div(std[j])
        end

        -- Forward pass and save the results
        local output = model:forward(testimg:cuda())
        if torch.type(output) == 'table' then -- case when the network is multi-out network
            output = output[4]
        end


        -- compute PCK
        local gt_jnt_hmap   = testjsc[{ {}, {1,opt.nJoints}, {}, {} }]
        local pred_jnt_hmap = output[{ {}, {1,opt.nJoints}, {}, {} }]
        local gt_jnt,occ   = find_peak(gt_jnt_hmap)
        local pred_jnt     = find_peak(pred_jnt_hmap)
        local pcks = torch.Tensor(numnormscalor)
        for j=1,numnormscalor do 
            local normscalor = (j-1)*0.05
            pcks[j] = comp_PCK(gt_jnt, pred_jnt, occ, normscalor)
        end
        pck_all[{ {i}, {} }] = pcks

        -- save output
        local savefile
        if testsettype == 'rTest' then
            savefile = savedir..string.format('/jsc_pred_frm%04d.mat', loader:fetch_framenumber(i))
        elseif testsettype == 'sTest' then
            savefile = savedir..string.format('/jsc_pred_frm%04d.mat', i)
        end
        --matio.save(savefile, output:float())
    end
    local pck_avg = torch.mean(pck_all,1)
    print(string.format('#model: %d | avgPCK(%%) (all images): ', mNum))
    for i=1,numnormscalor do
        print(string.format('%.2f ', pck_avg[1][i]))
    end

end

-- Delete the directory created unnecessarily
sys.fexecute("rm -r "..opt.save)

