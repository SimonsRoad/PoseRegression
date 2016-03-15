--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local defaultDir = '../save/'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Let me hit ECCV!!!')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache',
               defaultDir,
               'subdirectory in which to save/log experiments')
    --cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',      'cudnn', 'Options: cudnn | ccn2 | cunn')

	------------- Task options ------------------------
	cmd:option('-t',		   'PR_fcn', 'The name of task')

    ------------- Data options ------------------------
    cmd:option('-nDonkeys',           3, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-txtpos',      '../data/rendout/tmp_y144_x256_aug/lists/pos.txt',  'pos list')
    cmd:option('-txtjsdc',     '../data/rendout/tmp_y144_x256_aug/lists/jsdc.txt', 'jsdc list')
    cmd:option('-nTrainData',    300000,    'number of train data')
    cmd:option('-nTestData',       2000,    'number of test data')
    cmd:option('-W',                 64,    'image width')
    cmd:option('-H',                128,    'image height')

    ------------- Training options --------------------
    cmd:option('-nEpochs',         100,   'Number of total epochs to run')
    cmd:option('-epochSize',      6250,   'Number of batches per epoch')
    cmd:option('-epochNumber',       5,   'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',        48,   'mini-batch size (1 = pure stochastic)')

    ---------- Optimization options ----------------------
    cmd:option('-LR',    		   0.001, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    --cmd:option('-netType',     'alexnetowtbn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
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
