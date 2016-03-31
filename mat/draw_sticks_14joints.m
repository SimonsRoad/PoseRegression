function draw_sticks_14joints(joints, nJoints, style)
assert(size(joints,1) == nJoints)

htop = joints(1,:);
hbot = joints(2,:);
lsho = joints(3,:);
lelb = joints(4,:);
lwr = joints(5,:);
lhip = joints(6,:);
lkne = joints(7,:);
lank = joints(8,:);
rsho = joints(9,:);
relb = joints(10,:);
rwr = joints(11,:);
rhip = joints(12,:);
rkne = joints(13,:);
rank = joints(14,:);

sticks{1} = [htop; hbot];
sticks{2} = [hbot; lsho];
sticks{3} = [lsho; lelb];
sticks{4} = [lelb; lwr];
sticks{5} = [lsho; lhip]; 
sticks{6} = [lhip; lkne];
sticks{7} = [lkne; lank]; 
sticks{8} = [hbot; rsho];
sticks{9} = [rsho; relb]; 
sticks{10} = [relb; rwr];
sticks{11} = [rsho; rhip]; 
sticks{12} = [rhip; rkne]; 
sticks{13} = [rkne; rank];

for i = 1:numel(sticks)
    plot(sticks{i}(:,1)', sticks{i}(:,2)', style);
end

end