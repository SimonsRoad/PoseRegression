--[[
--test_pd.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


local matio = require 'matio'
require 'optim'
require 'cudnn'
require 'cunn';
paths.dofile('util.lua')
paths.dofile('datafromlist.lua')
paths.dofile('compute_distance.lua')
paths.dofile('misc_utils.lua')


-- 0. settings
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')

nTestData = 27


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/testimages.txt'}

testset_data = mydataloader:get(1, nTestData)
testset_label = torch.Tensor(nTestData):fill(1)
testset = {data = testset_data, label = testset_label}

print (testset)


-- *change data to cuda 
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()

-- load existing model
model = torch.load(modelSaved)

-- load mean and stdv
mean = matio.load('meanstd/meanForPD.mat')
stdv = matio.load('meanstd/stdvForPD.mat')


-- 2. test the network
--
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean.x[i][1])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv.x[i][1])
end

--test error
correct = 0
idx_incorrect = {}
for i=1, testset.label:size(1) do
	local groundtruth = testset.label[i]
	local prediction = model:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)

	if groundtruth == indices[1] then
		correct = correct + 1
	else
		table.insert(idx_incorrect, i)
	end
end
print ('(test) accuracy: ' .. 100*correct/testset.label:size(1) .. '% (' .. correct .. ' out of     ' .. testset.label:size(1) .. ')')
print('incorrect sample idx: ');print(idx_incorrect)


