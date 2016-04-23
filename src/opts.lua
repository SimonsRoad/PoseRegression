--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--
-----------------------------------------------------
-- Modified by Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--
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
    cmd:option('-txtimg',      '../data/rendout/anc_y185_x57/lists/img_pos.txt',  'img list')
    cmd:option('-txtjsc',      '../data/rendout/anc_y185_x57/lists/jsc_pos.txt',  'jsc list')
    cmd:option('-nTrainData',     50000,  'number of train data')
    cmd:option('-nTestData',        500,  'number of test data')
    cmd:option('-y',                185,  'x coordinate of anchor location')
    cmd:option('-x',                 57,  'y coordinate of anchor location')
    cmd:option('-W',                 82,  'image width')
    cmd:option('-H',                118,  'image height')
    cmd:option('-W_jsc',             82,  'jsc width')
    cmd:option('-H_jsc',            118,  'jsc height')
    cmd:option('-nJoints',           27,  'number of joints')
    cmd:option('-nChOut',            29,  'number of joints')

    ------------- Training options --------------------
    cmd:option('-nEpochs',           20,   'Number of total epochs to run')
    cmd:option('-epochSize',       6250,   'Number of batches per epoch')
    cmd:option('-epochNumber',        1,   'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',          8,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-resume',        'none',   'Path to directory containing checkpoint')

    ---------- Optimization options ----------------------
    cmd:option('-LR',    	       0.01,   'learning rate ')
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
