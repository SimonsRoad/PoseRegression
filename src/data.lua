--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--
-- Modified by Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
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


