--[[
-- compute_meanstdv.lua
-- Namhoon Lee, RI, CMU (namhoonl@andrew.cmu.edu)
--]]


function compute_meanstdv(indices)

    local pos = loader_pos:load_img(indices, 3)

    -- compute mean and stdv from images
    local mean = {}
    local stdv = {}
    for i=1,3 do
        mean[i] = pos[{ {}, {i}, {}, {} }]:mean()
        stdv[i] = pos[{ {}, {i}, {}, {} }]:std()
    end
    return mean, stdv
end
