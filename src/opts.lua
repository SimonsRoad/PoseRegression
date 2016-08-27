----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedestrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- opts.lua
-- - defines options 
-- - This source code is originally created by Facebook, Inc.
----------------------------------------------------------------------

local M = { }

function M.parse(arg)
    local defaultDir = '../save/'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('NIPS')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-cache',
               defaultDir,
               'subdirectory in which to save/log experiments')
    cmd:option('-dataset', 'towncenter',  'currently only available dataset')
    --cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1,  'Default preferred GPU')
    cmd:option('-nGPU',               1,  'Number of GPUs to use by default')
    cmd:option('-backend',      'cudnn',  'Options: cudnn | ccn2 | cunn')
    cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')

	------------- Task options ------------------------
	cmd:option('-t',		   'PR_fcn',  'The name of task')

    ------------- Data options ------------------------
    cmd:option('-nDonkeys',           3,  '# of donkeys to initialize (data loading threads)')
    cmd:option('-txtimg',      '../data/rendout/anc_y138_x167/lists/img_pos.txt',  'img list')
    cmd:option('-txtjsc',      '../data/rendout/anc_y138_x167/lists/jsc_pos.txt',  'jsc list')
    cmd:option('-nTrainData',     49000,  'number of train data')
    cmd:option('-nTestData',        500,  'number of test data')
    cmd:option('-y',                138,  'y coordinate of anchor location')
    cmd:option('-x',                167,  'x coordinate of anchor location')
    cmd:option('-W',                 71,  'image width')
    cmd:option('-H',                102,  'image height')
    cmd:option('-W_jsc',             71,  'jsc width')
    cmd:option('-H_jsc',            102,  'jsc height')
    cmd:option('-nJoints',           27,  'number of joints')
    cmd:option('-nChOut',            28,  'number of joints')

    ------------- Training options --------------------
    cmd:option('-nEpochs',           35,   'Number of total epochs to run')
    cmd:option('-epochSize',       3063,   'Number of batches per epoch')
    cmd:option('-epochNumber',        1,   'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',         16,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-resume',        'none',   'Path to directory containing checkpoint')

    ---------- Optimization options ----------------------
    cmd:option('-LR',    	       0.005,   'learning rate ')
    cmd:option('-momentum',         0.9,   'momentum')
    cmd:option('-weightDecay',     5e-4,   'weight decay')

    ---------- Model options ----------------------------------
    cmd:option('-netType', 'testmodel2',   'Options: resnet, sposenet, cpm, testmodel')
    cmd:option('-retrain',       'none',   'provide path to model to retrain with')
    cmd:option('-optimState',    'none',   'provide path to an optimState to reload from')
    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput', 'false', 'Share gradInput tensors to reduce memory usage')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
	opt.save = paths.concat(opt.cache, opt.t)
    opt.save = paths.concat(opt.save,
                            cmd:string('option', opt,
                                       {retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, 't_' .. os.date():gsub(' ',''))
    return opt
end

return M
