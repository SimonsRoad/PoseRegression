----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedesetrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- donkey.lua
-- - This source code creates a data loader
----------------------------------------------------------------------

require 'image'
paths.dofile('datanew.lua')
--paths.dofile('util.lua')

loader = dataLoader{txtimg=opt.txtimg, txtjsc=opt.txtjsc}


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local meanstdCache = paths.concat(opt.cache, string.format('meanstdCache/y%d_x%d.t7',opt.y,opt.x))

if paths.filep(meanstdCache) then
    local meanstd = torch.load(meanstdCache)
    mean = meanstd.mean
    std  = meanstd.std
    print('Loaded mean and std from cache.')
else
    print('Computing meanstd..')
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
    print('Saved meanstd into file.. Time: ', tm:time().real)
end 

