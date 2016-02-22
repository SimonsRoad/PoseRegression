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


-- 0. settings
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')


-- 1. load and normalize data
-- 
testset = matio.load('../mat/dataset/testdata.mat')
trainset = matio.load('../mat/dataset/traindata.mat')

-- convert label if it's fcn
--testset.label = convert_labels_to_fcnLabels(testset.label)

print (testset)
print (trainset)
--assert(testset.label:size(1) == 2000); 
--assert(testset.label:size(2) == nJoints)
--assert(testset.label:size(3) == 32)


-- normalization
mean = matio.load('meanstd/meanForPD.mat')
stdv = matio.load('meanstd/stdvForPD.mat')
for i=1,3 do
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean.x[i][1])
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv.x[i][1])
	testset.data[{ {}, {i}, {}, {} }]:add(-mean.x[i][1])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv.x[i][1])
end

-- *change data to cuda 
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()


-- load saved model
model = torch.load(modelSaved)
model:evaluate()

-- 2. Test
evaluate(testset, 'test', 'testdir')
evaluate(trainset, 'train', 'testdir')

