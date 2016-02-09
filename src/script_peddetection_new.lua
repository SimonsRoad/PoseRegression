--[[
--script_peddetection_new.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


require 'optim'
require 'cudnn'
require 'cunn'
paths.dofile('util.lua')
paths.dofile('datafromlist.lua')
paths.dofile('create_network.lua')

paths.dofile('save_results.lua')


-- 0. settings
task = 'PedDetection'
modelNumber = 1

nPool_pos = 13344
nPool_neg = 10000
nTrainData_pos = 5000
nTrainData_neg = 5000
nTestData_pos = 1000
nTestData_neg = 1000


-- 1. load and normalize data
--
-- data loader 
mydataloader_pos = dataLoader{filename = '../data/lists/pos.txt'}
mydataloader_neg = dataLoader{filename = '../data/lists/neg.txt'}

-- indexing. This should never change again while running the code 
idx_pool_pos = torch.randperm(nPool_pos)
idx_pool_neg = torch.randperm(nPool_neg)
idx_train_pos = idx_pool_pos:narrow(1,1,nTrainData_pos)
idx_train_neg = idx_pool_neg:narrow(1,1,nTrainData_neg)
idx_test_pos = idx_pool_pos:narrow(1,nTrainData_pos+1,nTestData_pos)
idx_test_neg = idx_pool_neg:narrow(1,nTrainData_neg+1,nTestData_neg)

trainset_pos = mydataloader_pos:get_randomly_indices(idx_train_pos)
trainset_neg = mydataloader_neg:get_randomly_indices(idx_train_neg)
trainset_data = torch.cat(trainset_pos, trainset_neg, 1)
trainset_label = torch.cat(torch.ones(trainset_pos:size(1)), torch.zeros(trainset_neg:size(1)))
trainset = {data = trainset_data, label = trainset_label} 

testset_pos = mydataloader_pos:get_randomly_indices(idx_test_pos)
testset_neg = mydataloader_neg:get_randomly_indices(idx_test_neg)
testset_data = torch.cat(testset_pos, testset_neg, 1)
testset_label= torch.cat(torch.ones(testset_pos:size(1)), torch.zeros(testset_neg:size(1)))
testset = {data = testset_data, label = testset_label} 

print (trainset)
print (testset)

setmetatable(trainset,
{__index = function(t,i)
	return {t.data[i], t.label[i]}
end}
);
function trainset:size()
	return self.data:size(1)
end

-- normalization
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])

	stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end


-- 2. network
--
--model = create_network_model1() -- 100% 
--model = create_network_model2() -- test: 64%, train: 65%
--model = create_network_model3() -- 96.85% 
model = create_network(modelNumber)
cudnn.convert(model, cudnn)

-- 3. loss function
-- 
criterion = nn.ClassNLLCriterion()


-- *change to cuda
model = model:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()


-- *optional
print(opt)
cutorch.setDevice(opt.GPU)


-- 4. trian the network
--
paths.dofile('train.lua')

epoch = opt.epochNumber
for i=1, opt.nEpochs do
	train()
	epoch = epoch + 1
end


-- 5. test the network
--
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

-- test error
correct = 0
for i=1, testset.label:size(1) do
	local groundtruth = testset.label[i]
	local prediction = model:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)

	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print ('(test) accuracy: ' .. 100*correct/testset.label:size(1) .. '% (' .. correct .. ' out of ' .. testset.label:size(1) .. ')')

-- train error
correct = 0
for i=1, trainset.label:size(1) do
	local groundtruth = trainset.label[i]
	local prediction = model:forward(trainset.data[i])
	local confidences, indices = torch.sort(prediction, true)

	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print ('(train)accuracy: ' .. 100*correct/trainset.label:size(1) .. '% (' .. correct .. ' out of ' .. trainset.label:size(1) .. ')')


