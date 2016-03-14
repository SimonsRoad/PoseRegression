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
paths.dofile('compute_meanstdv.lua')
paths.dofile('evaluate.lua')
paths.dofile('load_batch.lua')


-- 0. settings 
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')
print('Saving everything to: ' .. opt.save) 
os.execute('mkdir -p ' .. opt.save)


-- 1. load and normalize data
-- 
loader_pos = dataLoader{filename = '../data/rendout/tmp_y144_x256_aug/lists/pos.txt'}
loader_seg = dataLoader{filename = '../data/rendout/tmp_y144_x256_aug/lists/seg.txt'}
loader_dep = dataLoader{filename = '../data/rendout/tmp_y144_x256_aug/lists/dep.txt'}
loader_cen = dataLoader{filename = '../data/rendout/tmp_y144_x256_aug/lists/cen.txt'}
--loader_j27 = dataLoader{filename = '../data/rendout/tmp_y144_x256_aug/lists/j27.txt'}

nTrainData = 30000
nTestData  = 2000

mean, stdv = compute_meanstdv(torch.range(1,nTrainData, 10))

testset = load_batch(torch.range(nTrainData+1, nTrainData+nTestData))
--matio.save(paths.concat(opt.save, 'testdata.mat'), testset)
print(testset)
assert(testset.label:size(1) == nTestData); assert(testset.label:size(2) == 3)
--assert(testset.label:size(1) == nTestData); assert(testset.label:size(2) == 30)


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

	-- train and test
	train()
	test()

	-- evaluation
    --[[
	if epoch % 1 == 0 then
		evaluate(testset,  'test')
		evaluate(trainset, 'train')
		local t_eval = timer:time().real
		print(string.format('EP. [%d/%d] Time Analysys(s) [crop / fcn / main / eval]: %.2f / %.2f / %.2f / %.2f ',epoch,opt.nEpochs,t_crop, t_fcnlabel, t_main, t_eval))
        adf=adf+1
	end
    --]]
	epoch = epoch + 1
end




