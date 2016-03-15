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
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
--
-- Modified by Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--

loader = dataLoader{txtpos=opt.txtpos, txtjsdc=opt.txtjsdc}


-- channel-wise mean and std. Calculate or load them from disk later in the script.
indices = torch.randperm(opt.nTrainData)
local pos = loader:load_img(indices[{{1,math.min(100,opt.nTrainData)}}])
mean = {}
stdv = {}
for i=1,3 do
    mean[i] = pos[{ {}, {i}, {}, {} }]:mean()
    stdv[i] = pos[{ {}, {i}, {}, {} }]:std()
end

