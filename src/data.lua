----------------------------------------------------------------------
-- Copyright (c) 2016, Namhoon Lee <namhoonl@andrew.cmu.edu>
-- All rights reserved.
--
-- This file is part of NIPS'16 submission
-- Visual Compiler: Scene Description to Pedestrian Pose Estimation
-- N. Lee*, V. N. Boddeti*, K. M. Kitani, F. Beainy, and T. Kanade
--
-- data.lua
-- - This source code creates K threads for parallel data-loading
-- - This source code is originally created by Facebook, Inc.
----------------------------------------------------------------------

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
        local options = opt    
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
             opt = options
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end


