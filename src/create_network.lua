function create_network_model1()

	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,32,9,9))
	net:add(nn.ReLU(true))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(32,32,5,5))
	net:add(nn.ReLU(true))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(32,32,5,5))
	net:add(nn.ReLU(true))
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.View(32*12*4))

	net:add(nn.Linear(32*12*4, 512))
	net:add(nn.ReLU(true))
	
	net:add(nn.Linear(512, 512))
	net:add(nn.ReLU(true))
	
	net:add(nn.Linear(512, 2))

	net:add(nn.LogSoftMax())

	return net
end

function create_network_model2()

	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))

	net:add(nn.View(16*16*8))

	net:add(nn.LogSoftMax())

	return net
end

function create_network_model3()

	require 'nn';
	--require 'cudnn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))

	net:add(nn.View(16*32*16))

	net:add(nn.LogSoftMax())

	return net
end

function create_network_model4()

	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,5,5))
	net:add(nn.ReLU())
	
	net:add(nn.SpatialConvolution(16,16,5,5))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,5,5))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.View(16*28*12))

	net:add(nn.LogSoftMax())

	return net
end

function create_network_model5()
	
	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))

	net:add(nn.View(16*16*8))

	net:add(nn.Linear(16*16*8, 28))
	net:add(nn.ReLU())

	return net
end

function create_network_model6()
	
	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())

	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.View(16*16*8))

	net:add(nn.Linear(16*16*8, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 28))
	net:add(nn.ReLU())

	return net
end

function create_network_model7() -- same as 6, except for the output (upperbody 8 joints)
	
	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())

	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.View(16*16*8))

	net:add(nn.Linear(16*16*8, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 16))
	net:add(nn.ReLU())

	return net
end

function create_network_model8() -- same as 6, except for the output (lowerbody 6 joints)
	
	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())

	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	
	net:add(nn.View(16*16*8))

	net:add(nn.Linear(16*16*8, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 12))
	net:add(nn.ReLU())

	return net
end

function create_network_model9() -- same as 6; full-body, but larger output.
	
	require 'nn';

	local net = nn.Sequential()

	net:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.View(16*32*16))

	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(16*32*16, 512))
	net:add(nn.ReLU())

	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(512, 512))
	net:add(nn.ReLU())

	net:add(nn.Linear(512, 28))
	net:add(nn.ReLU())

	return net
end

function create_network_model10()		-- PR_multi
	
	require 'nn';

	--
	local feat = nn.Sequential()

	feat:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:cuda()

	feat = makeDataParallel(feat, opt.nGPU)
	
	--
	local regression_upper = nn.Sequential()
	regression_upper:add(nn.View(16*32*16))
	regression_upper:add(nn.Dropout(0.5))
	regression_upper:add(nn.Linear(16*32*16, 16))
	regression_upper:cuda()

	--
	local regression_lower = nn.Sequential()
	regression_lower:add(nn.View(16*32*16))
	regression_lower:add(nn.Dropout(0.5))
	regression_lower:add(nn.Linear(16*32*16, 12))
	regression_lower:cuda()

	--
	local tasks = nn.ConcatTable()
	tasks:add(regression_upper)
	tasks:add(regression_lower)
	tasks:cuda()

	--
	local model = nn.Sequential():add(feat):add(tasks)
	model:cuda()

	return model

end

function create_network_model11()			-- spatial filter; large output
	
	require 'nn';

	--
	local feat = nn.Sequential()

	feat:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))

	feat:cuda()
	feat = makeDataParallel(feat, opt.nGPU)
	
	--
	local regression = nn.Sequential()
	regression:add(nn.View(16*32*16))

	regression:add(nn.Dropout(0.5))
	regression:add(nn.Linear(16*32*16, 2688))
	regression:add(nn.ReLU())

	regression:add(nn.Linear(2688, 2688))

	regression:cuda()

	--
	local model = nn.Sequential():add(feat):add(regression)

	return model

end

function create_network_model12()		-- PR_filt_struct
	
	require 'nn';

	--
	local feat = nn.Sequential()

	feat:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:cuda()

	feat = makeDataParallel(feat, opt.nGPU)
	
	local nOutFromFeat = 16*32*16

	-- 
	local htop = nn.Sequential()
	htop:add(nn.View(nOutFromFeat))
	htop:add(nn.Dropout(0.5))
	htop:add(nn.Linear(nOutFromFeat, W+H))
	htop:cuda()
	-- 
	local hbot = nn.Sequential()
	hbot:add(nn.View(nOutFromFeat))
	hbot:add(nn.Dropout(0.5))
	hbot:add(nn.Linear(nOutFromFeat, W+H))
	hbot:cuda()
	-- 
	local lsho = nn.Sequential()
	lsho:add(nn.View(nOutFromFeat))
	lsho:add(nn.Dropout(0.5))
	lsho:add(nn.Linear(nOutFromFeat, W+H))
	lsho:cuda()
	-- 
	local lelb = nn.Sequential()
	lelb:add(nn.View(nOutFromFeat))
	lelb:add(nn.Dropout(0.5))
	lelb:add(nn.Linear(nOutFromFeat, W+H))
	lelb:cuda()
	-- 
	local lwr = nn.Sequential()
	lwr:add(nn.View(nOutFromFeat))
	lwr:add(nn.Dropout(0.5))
	lwr:add(nn.Linear(nOutFromFeat, W+H))
	lwr:cuda()
	-- 
	local lhip = nn.Sequential()
	lhip:add(nn.View(nOutFromFeat))
	lhip:add(nn.Dropout(0.5))
	lhip:add(nn.Linear(nOutFromFeat, W+H))
	lhip:cuda()
	-- 
	local lkne = nn.Sequential()
	lkne:add(nn.View(nOutFromFeat))
	lkne:add(nn.Dropout(0.5))
	lkne:add(nn.Linear(nOutFromFeat, W+H))
	lkne:cuda()
	-- 
	local lank = nn.Sequential()
	lank:add(nn.View(nOutFromFeat))
	lank:add(nn.Dropout(0.5))
	lank:add(nn.Linear(nOutFromFeat, W+H))
	lank:cuda()
	-- 
	local rsho = nn.Sequential()
	rsho:add(nn.View(nOutFromFeat))
	rsho:add(nn.Dropout(0.5))
	rsho:add(nn.Linear(nOutFromFeat, W+H))
	rsho:cuda()
	-- 
	local relb = nn.Sequential()
	relb:add(nn.View(nOutFromFeat))
	relb:add(nn.Dropout(0.5))
	relb:add(nn.Linear(nOutFromFeat, W+H))
	relb:cuda()
	-- 
	local rwr = nn.Sequential()
	rwr:add(nn.View(nOutFromFeat))
	rwr:add(nn.Dropout(0.5))
	rwr:add(nn.Linear(nOutFromFeat, W+H))
	rwr:cuda()
	-- 
	local rhip = nn.Sequential()
	rhip:add(nn.View(nOutFromFeat))
	rhip:add(nn.Dropout(0.5))
	rhip:add(nn.Linear(nOutFromFeat, W+H))
	rhip:cuda()
	-- 
	local rkne = nn.Sequential()
	rkne:add(nn.View(nOutFromFeat))
	rkne:add(nn.Dropout(0.5))
	rkne:add(nn.Linear(nOutFromFeat, W+H))
	rkne:cuda()
	-- 
	local rank = nn.Sequential()
	rank:add(nn.View(nOutFromFeat))
	rank:add(nn.Dropout(0.5))
	rank:add(nn.Linear(nOutFromFeat, W+H))
	rank:cuda()

	--
	local tasks = nn.ConcatTable()
	tasks:add(htop)
	tasks:add(hbot)
	tasks:add(lsho)
	tasks:add(lelb)
	tasks:add(lwr)
	tasks:add(lhip)
	tasks:add(lkne)
	tasks:add(lank)
	tasks:add(rsho)
	tasks:add(relb)
	tasks:add(rwr)
	tasks:add(rhip)
	tasks:add(rkne)
	tasks:add(rank)
	tasks:cuda()

	--
	local model = nn.Sequential():add(feat):add(tasks)

	return model
	
end


function create_network(modelNumber)
	local func = 'create_network_model' .. modelNumber
	local net = _G[func]()
	return net
end

































