--[[
-- load_batch.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]


function load_batch(indices)

    -- mix indices
    local order = torch.randperm(indices:size(1))
    local indices_mix = indices:index(1,order:long())

    -- load all
    local pos = loader_pos:load_img(indices_mix, 3)
    local seg = loader_seg:load_img(indices_mix, 1)
    local dep = loader_dep:load_img(indices_mix, 1)
    local cen = loader_cen:load_img(indices_mix, 1)
    --local j27 = loader_j27:load_j27(indices_mix)

    -- normalize images (pos)
    for i=1,3 do
        pos[{ {}, {i}, {}, {} }]:add(-mean[i])
        pos[{ {}, {i}, {}, {} }]:div(stdv[i])
    end

    -- labels
    local labeltensor = torch.cat({seg, dep, cen}, 2)

    -- out
    local out = {data = pos, label = labeltensor}
    return out
end


