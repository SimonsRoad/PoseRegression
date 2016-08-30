require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'paths'
npy4th = require 'npy4th'

paths.dofile('../load_dataset_config.lua')


path_base = '/home/namhoon/develop/PoseRegression'
datasetname = 'towncenter'
load_dataset_config(datasetname)

for i=1,10 do
    print (string.format('processing %d ..',i))
    local fmodel = path_base .. 
        string.format('/save/PR_fcn/option/%s/clear_model_%d.t7', 
        datetime[i],bestmodel[i])
    local model = torch.load(fmodel)
    local fsave = string.format('../../visualize_filters/conv1_npy/conv1_weights_loc%d.npy',i)
    npy4th.savenpy(fsave, model:get(2).weight)
end

