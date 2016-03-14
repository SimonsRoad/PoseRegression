--[[
-- load_batch.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]


function load_batch(indices)

    -- mix indices
    local order = torch.randperm(indices:size(1))
    local indices_mix = indices:index(1,order:long())

    -- load all
    local pos  = loader_pos:load_img(indices_mix)
    local jsdc = loader_jsdc:load_jsdc(indices_mix)

    -- normalize images (pos)
    for i=1,3 do
        pos[{ {}, {i}, {}, {} }]:add(-mean[i])
        pos[{ {}, {i}, {}, {} }]:div(stdv[i])
    end

    -- out
    local out = {data = pos, label = jsdc}
    return out
end


