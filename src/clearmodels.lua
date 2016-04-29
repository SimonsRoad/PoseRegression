-- [[
-- clearmodels.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- ]]

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

pathtomodels = '../save/PR_fcn/option/t_MonApr2522:54:292016'

for epoch = 1,10 do
    print('epoch ' .. epoch)
	local model_raw = torch.load(paths.concat(pathtomodels, 'model_'..epoch..'.t7'))
	local model_clr = model_raw:clearState()

	-- sanity check
	local a = torch.rand(1,3,128,64):cuda()
	local out_raw = model_raw:cuda():forward(a)
	local out_clr = model_clr:cuda():forward(a)
    if torch.type(out_raw) == 'table' then
        print(torch.norm(out_raw[#out_raw])); print(torch.norm(out_clr[#out_clr]));
        assert(torch.norm(out_raw[#out_raw]) == torch.norm(out_clr[#out_clr]))
    else
	    print(torch.norm(out_raw)); print(torch.norm(out_clr));
	    assert(torch.norm(out_raw) == torch.norm(out_clr))
    end

	-- save
	torch.save(paths.concat(pathtomodels, 'clear_model_'..epoch..'.t7'), model_clr)
end

