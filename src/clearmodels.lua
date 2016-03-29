-- [[
-- clearmodels.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- ]]

require 'nn'
require 'cunn'
require 'cudnn'

pathtomodels = '../save/PR_fcn/option/t_SunMar2721:48:402016'

for epoch = 4,10 do
	local model_raw = torch.load(paths.concat(pathtomodels, 'model_'..epoch..'.t7'))
	local model_clr = model_raw:clearState()

	-- sanity check
	local a = torch.rand(3,3,128,64):cuda()
	local out_raw = model_raw:cuda():forward(a)
	local out_clr = model_clr:cuda():forward(a)
	print(torch.norm(out_raw)); print(torch.norm(out_clr));
	assert(torch.norm(out_raw) == torch.norm(out_clr))

	-- save
	torch.save(paths.concat(pathtomodels, 'clear_model_'..epoch..'.t7'), model_clr)
end

