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

function create_network_model10()
	
	require 'nn'
	
	local feat = nn.Sequential()
	feat:add(nn.SpatialConvolution(3,16,3,3,1,1,1,1))
	feat:add(nn.ReLU)
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU)
	feat:add(nn.SpatialMaxPooling(2,2,2,2))
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU)
	feat:add(nn.SpatialConvolution(16,16,3,3,1,1,1,1))
	feat:add(nn.ReLU)
	feat:add(nn.SpatialMaxPooling(2,2,2,2))

	local detection = nn.Sequential() 
	detection:add(nn.View(16*32*16))
	detection:add(nn.Linear(16*32*16, 2))

	local regression_upper = nn.Sequential()
	regression_upper:add(nn.View(16*32*16))
	regression_upper:add(nn.Linear(16*32*16,16))

	local regression_lower = nn.Sequential()
	regression_lower:add(nn.View(16*32*16))
	regression_lower:add(nn.Linear(16*32*16,12))

	local regression_full = nn.Sequential()
	regression_full:add(nn.View(16*32*16))
	regression_full:add(nn.Linear(16*32*16,28))


	local tasks = nn.Concat(1)
	tasks:add(regression_upper)
	tasks:add(regression_lower)

	local model = nn.Sequential():add(feat):add(tasks)

	return model
end


function create_network(modelNumber)
	local func = 'create_network_model' .. modelNumber
	local net = _G[func]()
	return net
end

































