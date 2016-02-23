--[[
--script_poseregression_fcn_new.lua
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
paths.dofile('convert_labels.lua')
paths.dofile('evaluate.lua')
paths.dofile('randomcrop.lua')


-- 0. settings 
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')
print('Saving everything to: ' .. opt.save) 
os.execute('mkdir -p ' .. opt.save)


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/pos.txt'}

nTrainData = 10000
nTestData = 2000

-- original data (before crop or normalize)
trainset_ori = mydataloader:load_original(torch.range(1,nTrainData))
testset_ori  = mydataloader:load_original(torch.range(nTrainData+1,nTrainData+nTestData))
print(trainset_ori)
print(testset_ori)

-- compute mean and stdv from original
mean = {}
stdv = {}
for i=1,3 do
	mean[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:mean()
	stdv[i] = trainset_ori.data[{ {}, {i}, {}, {} }]:std()
end

-- testset
testset = randomcrop(testset_ori)
testset.label = convert_labels_to_fcnLabels(testset.label)
matio.save(paths.concat(opt.save, 'testdata.mat'), testset)
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end
assert(testset.label:size(1) == nTestData); 
assert(testset.label:size(2) == nJoints)
assert(testset.label:size(3) == 32)
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
criterion = nn.MSECriterion()
criterion = criterion:cuda()


-- 4. Training
--
print(opt)

paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')

local timer = torch.Timer()

epoch = opt.epochNumber
for i=1, opt.nEpochs do
	timer:reset()

	-- random crop
	local trainset = randomcrop(trainset_ori)
	local t_crop = timer:time().real

	-- convert label to fcnlabel 
	trainset.label = convert_labels_to_fcnLabels(trainset.label)
	local t_fcnlabel = timer:time().real

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
	local t_main = timer:time().real

	-- evaluation
	if epoch == 50 then
		evaluate(testset,  'test')
		--evaluate(trainset, 'train')
		local t_eval = timer:time().real
		print(string.format('EP. [%d/%d] Time Analysys(s) [crop / fcn / main / eval]: %.2f / %.2f / %.2f / %.2f ',epoch,opt.nEpochs,t_crop, t_fcnlabel, t_main, t_eval))
	end

	epoch = epoch + 1
end




