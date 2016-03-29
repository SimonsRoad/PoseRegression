-- [[
-- clearmodels.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- ]]

require 'nn'
require 'cunn'
require 'cudnn'

pathtomodels = '../save/PR_fcn/option/t_Tue'
epoch = 1

local model_raw = paths.concat(pathtomodels, 'model_'..epoch..'.t7')
model_raw:cuda()

local model_clr = model_raw:clearState()

-- sanity check
local a = torch.rand(3,3,128,64):cuda()
out_raw = model_raw:forward(a)
out_clr = model_clr:forward(a)
assert(torch.norm(out_raw) == torch.norm(out_clr))

-- save
torch.save(paths.concat(pathtomodels, 'clear_model_'..epoch..'.t7', model_clr) 

