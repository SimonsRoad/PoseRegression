--[[
--script_poseregression.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


-- Dependencies
local matio = require 'matio'
require 'optim'
paths.dofile('datafromlist.lua')
paths.dofile('create_network.lua')
paths.dofile('compute_distance.lua')
paths.dofile('save_results.lua')
paths.dofile('misc_utils.lua')


-- 0. settings
task = 'PoseRegression'
--part = 'upperbody'   -- nJoints:8, modelNumber:7
--part = 'lowerbody'   -- nJoints:6, modelNumber:8
part = 'fullbody'   -- nJoints:14, modelNumber:9
nJoints = 14
modelNumber = 9

nPoolSize = 12000
nTrainData = 10000
nTestData = 2000

time = get_time()


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/pos.txt'}

idx_pool = torch.randperm(nPoolSize)
idx_train = idx_pool:narrow(1,1,nTrainData)
idx_test = idx_pool:narrow(1,nTrainData+1,nTestData)

trainset_sample = mydataloader:get_randomly_indices(idx_train)
trainset_label = mydataloader:get_label(part, idx_train)
trainset = {data = trainset_sample, label = trainset_label} 
print (trainset)

testset_sample = mydataloader:get_randomly_indices(idx_test)
testset_label = mydataloader:get_label(part, idx_test)
testset = {data = testset_sample, label = testset_label}
print (testset)

save_testdata(testset, part, 'testpurpose') 	-- save into mat file

assert(testset.label:size(1) == nTestData)
assert(testset.label:size(2) == nJoints*2)

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
model = create_network(modelNumber)

-- cudnn
require 'cudnn'
cudnn.convert(model, cudnn)


-- 3. loss function
-- 
criterion = nn.MSECriterion()


-- 4. trian the network
--
-- cuda setup
require 'cunn';
model = model:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()

-- trainer
trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = 100
trainer.learningRate = 0.005
trainer.learningRateDecay = 0.1
--trainer.weightDecay = 0.0005
--trainer.momentum = 0.9

-- START TRAINING
print("maxIteration: " .. trainer.maxIteration)
print("learningRate: " .. trainer.learningRate)
print("learningRateDecay: " .. trainer.learningRateDecay)
trainer:train(trainset)


-- 5. test the network
--
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

pred_save_te, errPerJoint_te, meanErrPerJoint_te = compute_distance(testset, nJoints)
pred_save_tr, errPerJoint_tr, meanErrPerJoint_tr = compute_distance(trainset, nJoints)


-- *SAVE
-- save prediction results 
save_prediction(pred_save_te, pred_save_tr, part, time)

-- save log
saveTable = {
	['time'] = time,
	['task'] = task,
	['part'] = part,
	['nJoints'] = nJoints,
	['nTrainData'] = nTrainData,
	['nTestData'] = nTestData,
	['modelNumber'] = modelNumber,
	['trainer.maxiteration'] = trainer.maxIteration,
	['trainer.learningRate'] = trainer.learningRate,
	['trainer.learningRateDecay'] = trainer.learningRateDecay,
	['meanErrPerJoint_test'] = meanErrPerJoint_te,
	['meanErrPerJoint_train'] = meanErrPerJoint_tr
}
print(saveTable)
save_log(saveTable, time)



