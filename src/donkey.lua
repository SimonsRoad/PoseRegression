--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
paths.dofile('datanew.lua')
--paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
--
-- Modified by Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--

loader = dataLoader{txtimg=opt.txtimg, txtjsdc=opt.txtjsdc}


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

if paths.filep(meanstdCache) then
    local meanstd = torch.load(meanstdCache)
    mean = meanstd.mean
    std  = meanstd.std
    print('Loaded mean and std from cache.')
else
    local tm = torch.Timer()
    local indices = torch.randperm(opt.nTrainData)
    local pos = loader:load_img(indices[{{1,math.min(10000,opt.nTrainData)}}])
    local mean = {}
    local std  = {}
    for i=1,3 do
        mean[i] = pos[{ {}, {i}, {}, {} }]:mean()
        std[i]  = pos[{ {}, {i}, {}, {} }]:std()
    end
    local cache = {}
    cache.mean = mean
    cache.std  = std
    torch.save(meanstdCache, cache)
    print('Time to estimate:', tm:time().real)
end 


