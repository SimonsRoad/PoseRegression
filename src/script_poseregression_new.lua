--[[
--script_poseregression_new.lua
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
paths.dofile('save_results.lua')
paths.dofile('misc_utils.lua')


-- 0. settings
if opt.t == 'PR_full' then
	part = 'fullbody'; nJoints = 14; modelNumber = 9;   -- nJoints:14, modelNumber:9
elseif opt.t == 'PR_upper' then
	part = 'upperbody'; nJoints = 8; modelNumber = 7;   -- nJoints:8, modelNumber:7
elseif opt.t == 'PR_lower' then
	part = 'lowerbody'; nJoints = 6; modelNumber = 8;   -- nJoints:6, modelNumber:8
else assert(false, 'invalid task!!') end
print(string.format('\n**Performing [%s] modelNumber: %d\n', opt.t, modelNumber))

nPoolSize = 13344
nTrainData = 10000
nTestData = 2000


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/pos.txt'}

idx_pool = torch.randperm(nPoolSize)
idx_train = idx_pool:narrow(1,1,nTrainData)
idx_test = idx_pool:narrow(1,nTrainData+1,nTestData)

trainset_data = mydataloader:get_randomly_indices(idx_train)
trainset_label = mydataloader:get_label(part, idx_train)
trainset = {data = trainset_data, label = trainset_label} 

testset_data = mydataloader:get_randomly_indices(idx_test)
testset_label = mydataloader:get_label(part, idx_test)
testset = {data = testset_data, label = testset_label}

--print (trainset)
--print (testset)
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
cudnn.convert(model, cudnn)


-- 3. loss function
-- 
criterion = nn.MSECriterion()


-- *change to cuda 
model = model:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()


-- *Optional
cutorch.setDevice(opt.GPU)
print(opt)
print('Saving everything to: ' .. opt.save)


-- 4. (NEW) TRAINING  
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

pred_save_te, errPerJoint_te, meanErrPerJoint_te = compute_distance_joint(testset, nJoints)
pred_save_tr, errPerJoint_tr, meanErrPerJoint_tr = compute_distance_joint(trainset, nJoints)
avgMSE_te = compute_distance_MSE(testset)
avgMSE_tr = compute_distance_MSE(trainset)

print(meanErrPerJoint_te)
print(meanErrPerJoint_tr)
print(string.format('avgMSE (test) : %.4f', avgMSE_te))
print(string.format('avgMSE (train): %.4f', avgMSE_tr))






