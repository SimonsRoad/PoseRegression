require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'paths'
require 'image'
npy4th = require 'npy4th'

paths.dofile('../load_dataset_config.lua')


path_base = '/home/namhoon/develop/PoseRegression'
datasetname = 'towncenter'
load_dataset_config(datasetname)

for i=1,10 do
    print (string.format('location [%d]',i))

    -- load model 
    local fmodel = path_base .. 
        string.format('/save/PR_fcn/option/%s/clear_model_%d.t7', 
        datetime[i],bestmodel[i])
    local model = torch.load(fmodel)
    
    -- load meanstd
    local meanstd = torch.load(path_base .. 
        string.format('/save/meanstdCache/y%d_x%d.t7',YX[i][1],YX[i][2]))

    -- run forward pass
    local fname_img = string.format('testset1/%d.jpg',i)
    local testimg = torch.Tensor(1,3,WH[i][2],WH[i][1])
    testimg[1] = image.load(fname_img,3,'float')
    for k=1,3 do
        testimg[{ {}, {}, {}, {} }]:add(-meanstd.mean[k])
        testimg[{ {}, {}, {}, {} }]:div(meanstd.std[k])
    end

    local output = model:forward(testimg:cuda())

    -- activation maps: the main purpose of this script
    print(model:get(2).output:size())
    print(model:get(2).output:size())
    print(model:get(2).output:size())
    adf=adf+1
    
    --local fsave = string.format('../../visualize_filters/conv1_npy/conv1_weights_loc%d.npy',i)
    --npy4th.savenpy(fsave, model:get(2).weight)
end

