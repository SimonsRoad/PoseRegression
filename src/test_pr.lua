--[[
--test_pr.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


local matio = require 'matio'
require 'optim'
require 'cudnn'
require 'cunn';
paths.dofile('util.lua')
paths.dofile('datafromlist.lua')
paths.dofile('compute_distance.lua')
paths.dofile('convert_labels.lua')
paths.dofile('evaluate.lua')
paths.dofile('randomcrop.lua')


-- 0. settings
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/pos.txt'}

nTrainData = 10000
nTestData  = 2000

trainset_ori = mydataloader:load_original(torch.range(1,nTrainData))
--testset_ori  = mydataloader:load_original(torch.range(nTrainData+1, nTrainData+nTestData))
print(trainset_ori)
--print(testset_ori)

-- compute mean and stdv from original
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:mean()
	stdv[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:std()
end

-- prepare testset (randomcrop + normalize)
--testset = randomcrop(testset_ori)
--matio.save(string.format('../save/testdir/testdata.mat'), testset)		-- save before normalize

-- load testset.. 
testset = matio.load(string.format('../save/testdir/testdata.mat'))



for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end
assert(testset.label:size(1) == nTestData); assert(testset.label:size(2) == nJoints*2)

-- change to Cuda
testset.data  = testset.data:cuda()
testset.label = testset.label:cuda()
--trainset.data = trainset.data:cuda()
--trainset.label = trainset.label:cuda()

-- convert label if it's fcn
--testset.label = convert_labels_to_fcnLabels(testset.label)
--assert(testset.label:size(1) == 2000); 
--assert(testset.label:size(2) == nJoints)
--assert(testset.label:size(3) == 32)

-- load saved model
model = torch.load(modelSaved)
model:evaluate()


-- 2. Test
evaluate(testset, 'test', 'testdir')
--evaluate(trainset, 'train', 'testdir')

