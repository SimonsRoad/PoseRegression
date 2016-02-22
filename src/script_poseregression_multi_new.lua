--[[
--script_poseregression_multi_new.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


local matio = require 'matio'
require 'optim'
require 'cudnn'
require 'cunn';
paths.dofile('util.lua')
paths.dofile('datafromlist.lua')
paths.dofile('create_network.lua')
paths.dofile('compute_distance.lua')
paths.dofile('evaluate.lua')
paths.dofile('randomcrop.lua')


-- 0. settings
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')


-- 1. load original data
-- 
mydataloader = dataLoader{filename = '../data/lists/pos.txt'}

nTrainData = 100
nTestData  = 20

trainset_ori = mydataloader:load_original(torch.range(1,nTrainData))
testset_ori  = mydataloader:load_original(torch.range(nTrainData+1, nTrainData+nTestData))
print(trainset_ori)
print(testset_ori)

-- compute mean and stdv from original 
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:mean()
	stdv[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:std()
end

-- prepare testset (randomcrop + normalize)
testset = randomcrop(testset_ori)
assert(testset.label:size(1) == nTestData); assert(testset.label:size(2) == nJoints*2)
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end
testset.data  = testset.data:cuda()
testset.label = testset.label:cuda()


-- 2. network
--
if opt.retrain ~= 'none' then
	assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
	print('Loading model from file: ' .. opt.retrain)
	model = loadDataParallel(opt.retrain, opt.nGPU)
else
	model = create_network(modelNumber)
	cudnn.convert(model, cudnn)
end


-- 3. loss function
-- 
criterion = nn.ParallelCriterion():add(nn.MSECriterion(), 8/14):add(nn.MSECriterion(), 6/14)
criterion = criterion:cuda()


-- 4. Training 
--
print(opt)
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('train_multi.lua')
paths.dofile('test_multi.lua')

local timer = torch.Timer()

epoch = opt.epochNumber
for i=1, opt.nEpochs do
	timer:reset()

	-- random crop
	local trainset = randomcrop(trainset_ori)

	-- normalize
	for j = 1,3 do
		trainset.data[{ {}, {j}, {}, {} }]:add(-mean[j])
		trainset.data[{ {}, {j}, {}, {} }]:div(stdv[j])
	end

	trainset.data  = trainset.data:cuda()
	trainset.label = trainset.label:cuda()
	
	-- train and test
	train(trainset)
	test()

	-- evaluation
	if epoch % 50 == 0 then
		evaluate(testset,  'test')
		evaluate(trainset, 'train')
	end

	print(string.format('EP. [%d/%d] (Total) Time(s): %.2f',epoch,opt.nEpochs,timer:time().real))
	epoch = epoch + 1
end


