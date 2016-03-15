--[[
--script_poseregression_fcn_new.lua
--Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

local matio = require 'matio'
require 'optim'
require 'cudnn'
require 'cunn';
paths.dofile('util.lua')
paths.dofile('create_network.lua')
paths.dofile('compute_distance.lua')
paths.dofile('evaluate.lua')
paths.dofile('load_batch.lua')
paths.dofile('eval_jsdc.lua')


-- 0. settings 
cutorch.setDevice(opt.GPU)
paths.dofile('load_settings.lua')
print('Saving everything to: ' .. opt.save) 
os.execute('mkdir -p ' .. opt.save)


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

paths.dofile('data.lua')
paths.dofile('train_fcn.lua')
paths.dofile('test_fcn.lua')

pckLogger = optim.Logger(paths.concat(opt.save, 'pck_test.log'))

epoch = opt.epochNumber
for i=1,opt.nEpochs do

	-- train and test
	train()
	test()

	-- evaluation
	if epoch % 10 == 0 then
        eval_jsdc(testset)
		--evaluate(testset,  'test')
		--evaluate(trainset, 'train')
	end
	epoch = epoch + 1
end




