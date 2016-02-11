--[[
-- load_settings.m
-- Namhoon Lee, The Robotics Institute, Carnegie Mellon University
--]]


if opt.t == 'PR_full' then
	part = 'fullbody'; nJoints = 14; modelNumber = 9;   -- nJoints:14, modelNumber:9
	modelSaved = '/home/namhoon/develop/PoseRegression/save/PR_full/option,LR=0.005,nEpochs=100,t=PR_full/t_ThuFeb1101:06:102016/PR_full_model_100.t7'
	--modelSaved = '/home/namhoon/develop/PoseRegression/save/PR_full/option,nEpochs=100,t=PR_full/t_WedFeb1003:04:332016/PR_full_model_100.t7'
	
elseif opt.t == 'PR_upper' then
	part = 'upperbody'; nJoints = 8; modelNumber = 7;   -- nJoints:8, modelNumber:7
	modelSaved = '/home/namhoon/develop/PoseRegression/save/PR_upper/option,nEpochs=100,t=PR_upper/t_WedFeb1003:04:222016/PR_upper_model_100.t7'

elseif opt.t == 'PR_lower' then
	part = 'lowerbody'; nJoints = 6; modelNumber = 8;   -- nJoints:6, modelNumber:8
	modelSaved = '/home/namhoon/develop/PoseRegression/save/PR_lower/option,t=PR_lower/t_ThuFeb1101:07:122016/PR_lower_model_50.t7'
	--modelSaved = '/home/namhoon/develop/PoseRegression/save/PR_lower/option,t=PR_lower/t_WedFeb1009:11:262016/PR_lower_model_50.t7'
	
elseif opt.t == 'PD' then
	modelNumber = 1
	modelSaved = '/home/namhoon/develop/PoseRegression/save/PD/option,t=PD/t_WedFeb1003:04:132016/PD_model_100.t' 
	
else 
	assert(false, 'invalid task!!') 
end


print(string.format('\n**Performaing [%s] modelNumber: %d\n', opt.t, modelNumber))
