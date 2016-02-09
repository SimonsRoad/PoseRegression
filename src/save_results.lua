require 'misc_utils.lua'
local matio = require 'matio'

function save_log(table, time)
	local saveFile = ('../log/'..table['task']..'_'..table['part']..'_'..time..'.log')

	local f = assert(io.open(saveFile, 'w'))
	for k,v in pairs(table) do
		f:write(k)
		f:write(': ')
		f:write(v)
		f:write(' \n')
	end
	f:close()
end

function save_prediction(pred_save_te, pred_save_tr, part, time)
	-- save predictions for test, train data
	local saveFile_te = ('../mat/pred_te_'..part..'_'..time..'.mat') 
	local saveFile_tr = ('../mat/pred_tr_'..part..'_'..time..'.mat') 
	matio.save((saveFile_te), {label_pred_te = pred_save_te})
	matio.save((saveFile_tr), {label_pred_tr = pred_save_tr})
end

function save_testdata(testset, part, time)
	-- save testset for visualization
	-- I need this since the test data is randomized
	local saveFile = ('../mat/testdata_'..part..'_'..time..'.mat')
	testset.data = testset.data:float()
	testset.label = testset.label:float()
	matio.save((saveFile), testset)
end

function save_tmp(testset)
	local saveFile = ('../tmp/testset.mat')
	testset.data = testset.data:float()
	testset.label = testset.label:float()
	matio.save((saveFile), testset)
end
