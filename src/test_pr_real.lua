--[[
--test_pr_real.lua
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
nTestData  = 27

trainset_ori = mydataloader:load_original(torch.range(1,nTrainData))
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:mean()
	stdv[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:std()
end


testdataloader = dataLoader{filename = '../data/lists/real.txt'}
indices = torch.range(1,27)
testset_data = testdataloader:get_randomly_indices(indices)
testset_label = testdataloader:get_label_fortest(indices, '../data/cropped')
testset = {data = testset_data, label = testset_label}
print(testset)
matio.save('../save/testdir/testdata.mat', testset)

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

