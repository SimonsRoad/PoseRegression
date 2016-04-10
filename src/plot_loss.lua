-- [[
-- plot_loss.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
-- ]

require 'paths'
require 'gnuplot'



-- load file to read; train and test loss (batch)
dirtofile = '../../../'
fTrain = paths.concat(dirtofile, 'train_batch2.log')
--fTest  = paths.concat(dirtofile, 'test_batch.log')

nValTrain = tonumber(sys.fexecute("cat " .. fTrain .. "| wc -l"))-1
--nValTest  = tonumber(sys.fexecute("cat " .. fTest  .. "| wc -l"))-1
--assert(nValTrain == nValTest)
x = torch.range(1,nValTrain)
yTrain = torch.Tensor(nValTrain)
yTest  = torch.Tensor(nValTrain)


-- Train loss 
local file = assert(io.open(fTrain, "r"))
local count = 1
for line in file:lines() do
    loss = tonumber(line)
    if loss ~= nil then 
        yTrain[count] = loss
        count = count + 1
    end
end

gnuplot.plot(x, yTrain, '-')
