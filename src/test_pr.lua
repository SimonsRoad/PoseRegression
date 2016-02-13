--[[
--script_poseregression_multitask.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


local matio = require 'matio'
require 'optim'
require 'cudnn'
require 'cunn';
paths.dofile('util.lua')
paths.dofile('datafromlist.lua')
paths.dofile('compute_distance.lua')
paths.dofile('misc_utils.lua')


-- 0. settings
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')

nTestData = 1
pathtojoints = '/home/namhoon/Downloads/data_towncenter/frames_out'


-- 1. load and normalize data
-- 
mydataloader = dataLoader{filename = '../data/lists/testcrop.txt'}

indices = torch.range(1,nTestData)
testset_data = mydataloader:get_randomly_indices(indices)
testset_label = mydataloader:get_label_fortest(indices, pathtojoints)
testset = {data = testset_data, label = testset_label}

print (testset)
assert(testset.label:size(1) == nTestData); assert(testset.label:size(2) == nJoints*2)


-- save testset into .mat file for visualization
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)
matio.save(paths.concat(opt.save,string.format('testdata_%s.mat', opt.t)), testset)


-- *change data to cuda 
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()


-- *Optional
print(opt)
print(model)


-- load existing model
model = torch.load(modelSaved)

-- load mean and stdv
mean = matio.load('meanstd/meanForPD.mat')
stdv = matio.load('meanstd/stdvForPD.mat')


-- test the network
--
for i=1,3 do
	testset.data[{ {}, {i}, {}, {} }]:add(-mean.x[i][1])
	testset.data[{ {}, {i}, {}, {} }]:div(stdv.x[i][1])
end

PCP_te = compute_PCP(testset)
print(string.format('PCP (test) :   %.2f(%%)', PCP_te))

pred_save_te, errPerJoint_te, meanErrPerJoint_te = compute_distance_joint(testset, nJoints)
print(string.format('meanErrPerJoint (test) :   %.4f', meanErrPerJoint_te))

avgMSE_te = compute_distance_MSE(testset)
print(string.format('avgMSE (test) : %.4f', avgMSE_te))

-- To check the results on images, save prediction outputs into .mat file
matio.save(paths.concat(opt.save,string.format('pred_te_%s.mat', opt.t)), pred_save_te)



