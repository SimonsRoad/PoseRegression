--[[
testTorch.lua
Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]

-- 1. load and normalize data
-- 2. define neural network
-- 3. define loss function
-- 4. train network on training data
-- 5. test network on test data


-- 1. load and normalize data
--
--os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
--os.execute('unzip cifar10torchsmall.zip')
trainset = torch.load('cifar/cifar10-train.t7')
testset = torch.load('cifar/cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
print(#trainset.data)

print(classes[trainset.label[100]])

setmetatable(trainset,
{__index = function(t,i)
    return {t.data[i], t.label[i]}
end}
);
trainset.data = trainset.data:double()

function trainset:size()
    return self.data:size(1)
end

print(trainset:size())
print(trainset[33])

--
redChannel = trainset.data[{ {}, {1}, {}, {} }]
print(#redChannel)

--
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
	print('Channel ' .. i .. ', Mean: ' .. mean[i])
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])

	stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()
	print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i]) 
end

-- 2. time to define neural network
--
require 'nn';

net = nn.Sequential()

net:add(nn.SpatialConvolution(3,6,5,5))
net:add(nn.Sigmoid())
net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.Sigmoid())
net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.View(16*5*5))

net:add(nn.Linear(16*5*5, 120))
net:add(nn.Sigmoid())

net:add(nn.Linear(120, 84))
net:add(nn.Sigmoid())

net:add(nn.Linear(84, 10))

net:add(nn.LogSoftMax())


-- 3. define the loss function
--
criterion = nn.ClassNLLCriterion()


-- 4. Train the neural network
--
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)


-- 5. test the network
--
print(classes[testset.label[100]])

testset.data = testset.data:double()
for i=1,3 do 
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

horse = testset.data[100]
print(horse:mean(), horse:std())

print(classes[testset.label[100]])
predicted = net:forward(testset.data[100])
print(predicted:exp())

--
for i = 1, predicted:size(1) do
	print(classes[i], predicted[i])
end

correct = 0
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)

	if groundtruth ==indices[1] then
		correct = correct + 1
	end
end
print(correct, 100*correct/10000 .. ' % ')

--
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)

	if groundtruth == indices[1] then
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/1000 .. ' % ')
end


-- cunn: neural networks on GPUs using CUDA
require 'cunn';

net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)




















