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

	net:cuda()

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

	--
	local feat = nn.Sequential()

	feat:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	-- one more conv is added here
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	--
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:cuda()
	
	feat = makeDataParallel(feat, opt.nGPU)

	--
	local nOutFromFeat = 16*32*16
	local dFC = 1024

	--
	local regression = nn.Sequential()
	regression:add(nn.View(nOutFromFeat))
	regression:add(nn.Dropout(0.5))
	regression:add(nn.Linear(nOutFromFeat, dFC))
	regression:add(nn.ReLU())
	regression:add(nn.Dropout(0.5))
	regression:add(nn.Linear(dFC, dFC))
	regression:add(nn.ReLU())
	regression:add(nn.Linear(dFC, 28))
	regression:cuda()

	--
	local model = nn.Sequential():add(feat):add(regression)
	model:cuda()

	return model
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
	
	local nOutFromFeat = 16*32*16
	local dFC = 1024

	--
	local regression_upper = nn.Sequential()
	regression_upper:add(nn.View(nOutFromFeat))
	regression_upper:add(nn.Dropout(0.5))
	regression_upper:add(nn.Linear(nOutFromFeat, dFC))
	regression_upper:add(nn.ReLU())
	regression_upper:add(nn.Dropout(0.5))
	regression_upper:add(nn.Linear(dFC, dFC))
	regression_upper:add(nn.ReLU())
	regression_upper:add(nn.Linear(dFC, 16))
	regression_upper:cuda()

	--
	local regression_lower = nn.Sequential()
	regression_lower:add(nn.View(nOutFromFeat))
	regression_lower:add(nn.Dropout(0.5))
	regression_lower:add(nn.Linear(nOutFromFeat, dFC))
	regression_lower:add(nn.ReLU())
	regression_lower:add(nn.Dropout(0.5))
	regression_lower:add(nn.Linear(dFC, dFC))
	regression_lower:add(nn.ReLU())
	regression_lower:add(nn.Linear(dFC, 12))
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

function create_network_model11()			-- PR_torsolimbs 

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
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:cuda()

	feat = makeDataParallel(feat, opt.nGPU)
	
	local nOutFromFeat = 16*16*8
	local dFC = 1024

	--
	local torso = nn.Sequential()
	torso:add(nn.View(nOutFromFeat))
	torso:add(nn.Dropout(0.5))
	torso:add(nn.Linear(nOutFromFeat, nOutFromFeat))
	torso:add(nn.ReLU())
	torso:add(nn.Dropout(0.5))
	torso:add(nn.Linear(nOutFromFeat, dFC))
	torso:add(nn.ReLU())
	torso:add(nn.Linear(dFC, 12))
	torso:cuda()

	--
	local limbs = nn.Sequential()
	limbs:add(nn.View(nOutFromFeat))
	limbs:add(nn.Dropout(0.5))
	limbs:add(nn.Linear(nOutFromFeat, nOutFromFeat))
	limbs:add(nn.ReLU())
	limbs:add(nn.Dropout(0.5))
	limbs:add(nn.Linear(nOutFromFeat, dFC))
	limbs:add(nn.ReLU())
	limbs:add(nn.Linear(dFC, 16))
	limbs:cuda()

	--
	local tasks = nn.ConcatTable()
	tasks:add(torso)
	tasks:add(limbs)
	tasks:cuda()

	--
	local model = nn.Sequential():add(feat):add(tasks)
	model:cuda()

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
	local dFC = 1024

	-- 
	local htop = nn.Sequential()
	htop:add(nn.View(nOutFromFeat))
	htop:add(nn.Dropout(0.5))
	htop:add(nn.Linear(nOutFromFeat, dFC))
	htop:add(nn.ReLU())
	htop:add(nn.Dropout(0.5))
	htop:add(nn.Linear(dFC, dFC))
	htop:add(nn.ReLU())
	htop:add(nn.Linear(dFC, W+H))
	htop:cuda()
	-- 
	local hbot = nn.Sequential()
	hbot:add(nn.View(nOutFromFeat))
	hbot:add(nn.Dropout(0.5))
	hbot:add(nn.Linear(nOutFromFeat, dFC))
	hbot:add(nn.ReLU())
	hbot:add(nn.Dropout(0.5))
	hbot:add(nn.Linear(dFC, dFC))
	hbot:add(nn.ReLU())
	hbot:add(nn.Linear(dFC, W+H))
	hbot:cuda()
	-- 
	local lsho = nn.Sequential()
	lsho:add(nn.View(nOutFromFeat))
	lsho:add(nn.Dropout(0.5))
	lsho:add(nn.Linear(nOutFromFeat, dFC))
	lsho:add(nn.ReLU())
	lsho:add(nn.Dropout(0.5))
	lsho:add(nn.Linear(dFC, dFC))
	lsho:add(nn.ReLU())
	lsho:add(nn.Linear(dFC, W+H))
	lsho:cuda()
	-- 
	local lelb = nn.Sequential()
	lelb:add(nn.View(nOutFromFeat))
	lelb:add(nn.Dropout(0.5))
	lelb:add(nn.Linear(nOutFromFeat, dFC))
	lelb:add(nn.ReLU())
	lelb:add(nn.Dropout(0.5))
	lelb:add(nn.Linear(dFC, dFC))
	lelb:add(nn.ReLU())
	lelb:add(nn.Linear(dFC, W+H))
	lelb:cuda()
	-- 
	local lwr = nn.Sequential()
	lwr:add(nn.View(nOutFromFeat))
	lwr:add(nn.Dropout(0.5))
	lwr:add(nn.Linear(nOutFromFeat, dFC))
	lwr:add(nn.ReLU())
	lwr:add(nn.Dropout(0.5))
	lwr:add(nn.Linear(dFC, dFC))
	lwr:add(nn.ReLU())
	lwr:add(nn.Linear(dFC, W+H))
	lwr:cuda()
	-- 
	local lhip = nn.Sequential()
	lhip:add(nn.View(nOutFromFeat))
	lhip:add(nn.Dropout(0.5))
	lhip:add(nn.Linear(nOutFromFeat, dFC))
	lhip:add(nn.ReLU())
	lhip:add(nn.Dropout(0.5))
	lhip:add(nn.Linear(dFC, dFC))
	lhip:add(nn.ReLU())
	lhip:add(nn.Linear(dFC, W+H))
	lhip:cuda()
	-- 
	local lkne = nn.Sequential()
	lkne:add(nn.View(nOutFromFeat))
	lkne:add(nn.Dropout(0.5))
	lkne:add(nn.Linear(nOutFromFeat, dFC))
	lkne:add(nn.ReLU())
	lkne:add(nn.Dropout(0.5))
	lkne:add(nn.Linear(dFC, dFC))
	lkne:add(nn.ReLU())
	lkne:add(nn.Linear(dFC, W+H))
	lkne:cuda()
	-- 
	local lank = nn.Sequential()
	lank:add(nn.View(nOutFromFeat))
	lank:add(nn.Dropout(0.5))
	lank:add(nn.Linear(nOutFromFeat, dFC))
	lank:add(nn.ReLU())
	lank:add(nn.Dropout(0.5))
	lank:add(nn.Linear(dFC, dFC))
	lank:add(nn.ReLU())
	lank:add(nn.Linear(dFC, W+H))
	lank:cuda()
	-- 
	local rsho = nn.Sequential()
	rsho:add(nn.View(nOutFromFeat))
	rsho:add(nn.Dropout(0.5))
	rsho:add(nn.Linear(nOutFromFeat, dFC))
	rsho:add(nn.ReLU())
	rsho:add(nn.Dropout(0.5))
	rsho:add(nn.Linear(dFC, dFC))
	rsho:add(nn.ReLU())
	rsho:add(nn.Linear(dFC, W+H))
	rsho:cuda()
	-- 
	local relb = nn.Sequential()
	relb:add(nn.View(nOutFromFeat))
	relb:add(nn.Dropout(0.5))
	relb:add(nn.Linear(nOutFromFeat, dFC))
	relb:add(nn.ReLU())
	relb:add(nn.Dropout(0.5))
	relb:add(nn.Linear(dFC, dFC))
	relb:add(nn.ReLU())
	relb:add(nn.Linear(dFC, W+H))
	relb:cuda()
	-- 
	local rwr = nn.Sequential()
	rwr:add(nn.View(nOutFromFeat))
	rwr:add(nn.Dropout(0.5))
	rwr:add(nn.Linear(nOutFromFeat, dFC))
	rwr:add(nn.ReLU())
	rwr:add(nn.Dropout(0.5))
	rwr:add(nn.Linear(dFC, dFC))
	rwr:add(nn.ReLU())
	rwr:add(nn.Linear(dFC, W+H))
	rwr:cuda()
	-- 
	local rhip = nn.Sequential()
	rhip:add(nn.View(nOutFromFeat))
	rhip:add(nn.Dropout(0.5))
	rhip:add(nn.Linear(nOutFromFeat, dFC))
	rhip:add(nn.ReLU())
	rhip:add(nn.Dropout(0.5))
	rhip:add(nn.Linear(dFC, dFC))
	rhip:add(nn.ReLU())
	rhip:add(nn.Linear(dFC, W+H))
	rhip:cuda()
	-- 
	local rkne = nn.Sequential()
	rkne:add(nn.View(nOutFromFeat))
	rkne:add(nn.Dropout(0.5))
	rkne:add(nn.Linear(nOutFromFeat, dFC))
	rkne:add(nn.ReLU())
	rkne:add(nn.Dropout(0.5))
	rkne:add(nn.Linear(dFC, dFC))
	rkne:add(nn.ReLU())
	rkne:add(nn.Linear(dFC, W+H))
	rkne:cuda()
	-- 
	local rank = nn.Sequential()
	rank:add(nn.View(nOutFromFeat))
	rank:add(nn.Dropout(0.5))
	rank:add(nn.Linear(nOutFromFeat, dFC))
	rank:add(nn.ReLU())
	rank:add(nn.Dropout(0.5))
	rank:add(nn.Linear(dFC, dFC))
	rank:add(nn.ReLU())
	rank:add(nn.Linear(dFC, W+H))
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
	model:cuda()

	return model
	
end

function create_network_model13()		-- PR_eachjoint
	
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
	local dFC = 1024
	
	--
	local htop = nn.Sequential()
	htop:add(nn.View(nOutFromFeat))
	htop:add(nn.Dropout(0.5))
	htop:add(nn.Linear(nOutFromFeat, dFC))
	htop:add(nn.ReLU())
	htop:add(nn.Dropout(0.5))
	htop:add(nn.Linear(dFC, dFC))
	htop:add(nn.ReLU())
	htop:add(nn.Linear(dFC, 2))
	htop:cuda()
	-- 
	local hbot = nn.Sequential()
	hbot:add(nn.View(nOutFromFeat))
	hbot:add(nn.Dropout(0.5))
	hbot:add(nn.Linear(nOutFromFeat, dFC))
	hbot:add(nn.ReLU())
	hbot:add(nn.Dropout(0.5))
	hbot:add(nn.Linear(dFC, dFC))
	hbot:add(nn.ReLU())
	hbot:add(nn.Linear(dFC, 2))
	hbot:cuda()
	-- 
	local lsho = nn.Sequential()
	lsho:add(nn.View(nOutFromFeat))
	lsho:add(nn.Dropout(0.5))
	lsho:add(nn.Linear(nOutFromFeat, dFC))
	lsho:add(nn.ReLU())
	lsho:add(nn.Dropout(0.5))
	lsho:add(nn.Linear(dFC, dFC))
	lsho:add(nn.ReLU())
	lsho:add(nn.Linear(dFC, 2))
	lsho:cuda()
	-- 
	local lelb = nn.Sequential()
	lelb:add(nn.View(nOutFromFeat))
	lelb:add(nn.Dropout(0.5))
	lelb:add(nn.Linear(nOutFromFeat, dFC))
	lelb:add(nn.ReLU())
	lelb:add(nn.Dropout(0.5))
	lelb:add(nn.Linear(dFC, dFC))
	lelb:add(nn.ReLU())
	lelb:add(nn.Linear(dFC, 2))
	lelb:cuda()
	-- 
	local lwr = nn.Sequential()
	lwr:add(nn.View(nOutFromFeat))
	lwr:add(nn.Dropout(0.5))
	lwr:add(nn.Linear(nOutFromFeat, dFC))
	lwr:add(nn.ReLU())
	lwr:add(nn.Dropout(0.5))
	lwr:add(nn.Linear(dFC, dFC))
	lwr:add(nn.ReLU())
	lwr:add(nn.Linear(dFC, 2))
	lwr:cuda()
	-- 
	local lhip = nn.Sequential()
	lhip:add(nn.View(nOutFromFeat))
	lhip:add(nn.Dropout(0.5))
	lhip:add(nn.Linear(nOutFromFeat, dFC))
	lhip:add(nn.ReLU())
	lhip:add(nn.Dropout(0.5))
	lhip:add(nn.Linear(dFC, dFC))
	lhip:add(nn.ReLU())
	lhip:add(nn.Linear(dFC, 2))
	lhip:cuda()
	-- 
	local lkne = nn.Sequential()
	lkne:add(nn.View(nOutFromFeat))
	lkne:add(nn.Dropout(0.5))
	lkne:add(nn.Linear(nOutFromFeat, dFC))
	lkne:add(nn.ReLU())
	lkne:add(nn.Dropout(0.5))
	lkne:add(nn.Linear(dFC, dFC))
	lkne:add(nn.ReLU())
	lkne:add(nn.Linear(dFC, 2))
	lkne:cuda()
	-- 
	local lank = nn.Sequential()
	lank:add(nn.View(nOutFromFeat))
	lank:add(nn.Dropout(0.5))
	lank:add(nn.Linear(nOutFromFeat, dFC))
	lank:add(nn.ReLU())
	lank:add(nn.Dropout(0.5))
	lank:add(nn.Linear(dFC, dFC))
	lank:add(nn.ReLU())
	lank:add(nn.Linear(dFC, 2))
	lank:cuda()
	-- 
	local rsho = nn.Sequential()
	rsho:add(nn.View(nOutFromFeat))
	rsho:add(nn.Dropout(0.5))
	rsho:add(nn.Linear(nOutFromFeat, dFC))
	rsho:add(nn.ReLU())
	rsho:add(nn.Dropout(0.5))
	rsho:add(nn.Linear(dFC, dFC))
	rsho:add(nn.ReLU())
	rsho:add(nn.Linear(dFC, 2))
	rsho:cuda()
	-- 
	local relb = nn.Sequential()
	relb:add(nn.View(nOutFromFeat))
	relb:add(nn.Dropout(0.5))
	relb:add(nn.Linear(nOutFromFeat, dFC))
	relb:add(nn.ReLU())
	relb:add(nn.Dropout(0.5))
	relb:add(nn.Linear(dFC, dFC))
	relb:add(nn.ReLU())
	relb:add(nn.Linear(dFC, 2))
	relb:cuda()
	-- 
	local rwr = nn.Sequential()
	rwr:add(nn.View(nOutFromFeat))
	rwr:add(nn.Dropout(0.5))
	rwr:add(nn.Linear(nOutFromFeat, dFC))
	rwr:add(nn.ReLU())
	rwr:add(nn.Dropout(0.5))
	rwr:add(nn.Linear(dFC, dFC))
	rwr:add(nn.ReLU())
	rwr:add(nn.Linear(dFC, 2))
	rwr:cuda()
	-- 
	local rhip = nn.Sequential()
	rhip:add(nn.View(nOutFromFeat))
	rhip:add(nn.Dropout(0.5))
	rhip:add(nn.Linear(nOutFromFeat, dFC))
	rhip:add(nn.ReLU())
	rhip:add(nn.Dropout(0.5))
	rhip:add(nn.Linear(dFC, dFC))
	rhip:add(nn.ReLU())
	rhip:add(nn.Linear(dFC, 2))
	rhip:cuda()
	-- 
	local rkne = nn.Sequential()
	rkne:add(nn.View(nOutFromFeat))
	rkne:add(nn.Dropout(0.5))
	rkne:add(nn.Linear(nOutFromFeat, dFC))
	rkne:add(nn.ReLU())
	rkne:add(nn.Dropout(0.5))
	rkne:add(nn.Linear(dFC, dFC))
	rkne:add(nn.ReLU())
	rkne:add(nn.Linear(dFC, 2))
	rkne:cuda()
	-- 
	local rank = nn.Sequential()
	rank:add(nn.View(nOutFromFeat))
	rank:add(nn.Dropout(0.5))
	rank:add(nn.Linear(nOutFromFeat, dFC))
	rank:add(nn.ReLU())
	rank:add(nn.Dropout(0.5))
	rank:add(nn.Linear(dFC, dFC))
	rank:add(nn.ReLU())
	rank:add(nn.Linear(dFC, 2))
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
	model:cuda()

	return model

end

function create_network_model14() -- PR_fcn
	
	require 'nn';

	--
	local feat = nn.Sequential()

	feat:add(nn.SpatialConvolution(3,128,5,5,1,1,2,2))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))

	feat:add(nn.SpatialConvolution(128,128,5,5,1,1,2,2))
	feat:add(nn.ReLU())
	feat:add(nn.SpatialMaxPooling(2,2,2,2))

	feat:add(nn.SpatialConvolution(128,32,5,5,1,1,2,2))
	feat:add(nn.ReLU())

	feat:add(nn.SpatialConvolution(32,512,9,9,1,1,4,4))
	feat:add(nn.ReLU())
	--
	feat:add(nn.SpatialConvolution(512,512,1,1))
	feat:add(nn.ReLU())
	
	feat:add(nn.SpatialConvolution(512,14,1,1))
	feat:add(nn.ReLU())

	feat = makeDataParallel(feat, opt.nGPU)
	feat:cuda()

	return feat
end


function create_network(modelNumber)
	local func = 'create_network_model' .. modelNumber
	local net = _G[func]()
	return net
end

































