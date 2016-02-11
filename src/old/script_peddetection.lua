--[[
--script_peddetection.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


-- 1. load and normalize data
-- 
paths.dofile('datafromlist.lua')
paths.dofile('create_network.lua')

mydataloader_pos = dataLoader{filename = '../data/lists/pos.txt'}
mydataloader_neg = dataLoader{filename = '../data/lists/neg.txt'}
--trainset_sample_1 = mydataloader_pos:get(1,5000)
--trainset_sample_2 = mydataloader_neg:get(1,5000)
trainset_sample_1 = mydataloader_pos:get_randomly(10000,5000)
trainset_sample_2 = mydataloader_neg:get_randomly(9000,5000)
trainset_sample = torch.cat(trainset_sample_1, trainset_sample_2, 1)
testset_sample_1 = mydataloader_pos:get(10001, 11000)
testset_sample_2 = mydataloader_neg:get(9001, 10000)
testset_sample = torch.cat(testset_sample_1, testset_sample_2, 1)

label_train = torch.cat(torch.ones(trainset_sample_1:size(1)), torch.ones(trainset_sample_2:size(1))+1)
label_test = torch.cat(torch.ones(testset_sample_1:size(1)), torch.ones(testset_sample_2:size(1))+1)

trainset = {data = trainset_sample, label = label_train} 
testset = {data = testset_sample, label = label_test} 

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
--	print('Channel ' .. i .. ', Mean: ' .. mean[i])
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])

	stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()
--	print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end


-- 2. network
--
--net = create_network_model1() -- 100% 
--net = create_network_model2() -- test: 64%, train: 65%
--net = create_network_model3() -- 96.85% 
modelNumber = 1
model = create_network(modelNumber)

-- 3. loss function
-- 
criterion = nn.ClassNLLCriterion()


-- 4. trian the network
--
-- cuda setup
require 'cunn';
model = model:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
testset.data = testset.data:cuda()

-- trainer
trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = 60
trainer.learningRate = 0.01
trainer.learningRateDecay = 0.0001
trainer:train(trainset)


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
	--print ("[" .. i .. "th iteration] groundtruth: " .. groundtruth .. ", indices[1]: " .. indices[1])

	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print ('(test) accuracy: ' .. 100*correct/testset.label:size(1) .. '% (' .. correct .. ' out of ' .. testset.label:size(1) .. ')')

-- train error
correct = 0
for i=1, trainset.label:size(1) do
	local groundtruth = trainset.label[i]
	local prediction = net:forward(trainset.data[i])
	local confidences, indices = torch.sort(prediction, true)
	--print ("[" .. i .. "th iteration] groundtruth: " .. groundtruth .. ", indices[1]: " .. indices[1])

	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print ('(train)accuracy: ' .. 100*correct/trainset.label:size(1) .. '% (' .. correct .. ' out of ' .. trainset.label:size(1) .. ')')


print('configuration:')
print('modelNumber: ' .. modelNumber)
print('maxIteration: ' .. trainer.maxIteration)
print('learningRate: ' .. trainer.learningRate)
print('learningRateDecay: ' .. trainer.learningRateDecay)
