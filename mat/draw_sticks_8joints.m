function draw_sticks_8joints(joints, nJoints, style)
assert(size(joints,1) == nJoints)

htop = joints(1,:);
hbot = joints(2,:);
lsho = joints(3,:);
lelb = joints(4,:);
lwr = joints(5,:);
rsho = joints(6,:);
relb = joints(7,:);
rwr = joints(8,:);

sticks{1} = [htop; hbot];
sticks{2} = [hbot; lsho];
sticks{3} = [lsho; lelb];
sticks{4} = [lelb; lwr];
sticks{5} = [hbot; rsho];
sticks{6} = [rsho; relb];
sticks{7} = [relb; rwr];

for i = 1:numel(sticks)
    plot(sticks{i}(:,1)', sticks{i}(:,2)', style);
end

end
