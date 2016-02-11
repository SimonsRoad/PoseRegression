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


-- 0. settings + loading
paths.dofile('load_settings.lua')

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

--print (trainset); print (testset)
assert(testset.label:size(1) == nTestData);assert(testset.label:size(2) == nJoints*2)

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

-- save testset into .mat file for visualization 
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)
matio.save(paths.concat(opt.save,string.format('testdata_%s.mat', opt.t)), testset)


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


-- 4. (NEW) TRAINING  
--
TRAINING = false
if TRAINING then
	paths.dofile('train.lua')
	epoch = opt.epochNumber
	for i=1, opt.nEpochs do
		train()
		epoch = epoch + 1
	end
	model:evaluate()
else
	-- *load existing model
	model = torch.load(modelSaved)
end


-- 5. test the network
--
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

PCP_te = compute_PCP(testset)
PCP_tr = compute_PCP(trainset)
pred_save_te, errPerJoint_te, meanErrPerJoint_te = compute_distance_joint(testset, nJoints)
pred_save_tr, errPerJoint_tr, meanErrPerJoint_tr = compute_distance_joint(trainset, nJoints)
avgMSE_te = compute_distance_MSE(testset)
avgMSE_tr = compute_distance_MSE(trainset)

print(string.format('PCP (test) :   %.2f(%%)', PCP_te))
print(string.format('PCP (train):   %.2f(%%)', PCP_tr))
print(string.format('meanErrPerJoint (test) :   %.4f', meanErrPerJoint_te))
print(string.format('meanErrPerJoint (train):   %.4f', meanErrPerJoint_tr))
print(string.format('avgMSE (test) :   %.4f', avgMSE_te))
print(string.format('avgMSE (train):   %.4f', avgMSE_tr))

-- To check the results on images, save prediction outputs into .mat file
matio.save(paths.concat(opt.save,string.format('pred_te_%s.mat', opt.t)), pred_save_te)




