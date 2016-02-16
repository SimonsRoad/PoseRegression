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
	cmd:option('-t',		   'noname', 'The name of task')
    ------------- Data options ------------------------
    --cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    --cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    --cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    --cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         50,    'Number of total epochs to run')
    cmd:option('-epochSize',       157,   'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       64,    'mini-batch size (1 = pure stochastic)')
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
