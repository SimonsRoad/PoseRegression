function draw_sticks_joints(joints, nJoints, style)

assert(size(joints,1) == nJoints)

if nJoints == 14        % fullbody
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
elseif nJoints == 8     % upperbody
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
elseif nJoints == 6     % lowerbody
    lhip = joints(1,:);
    lkne = joints(2,:);
    lank = joints(3,:);
    rhip = joints(4,:);
    rkne = joints(5,:);
    rank = joints(6,:);
    sticks{1} = [lhip; lkne];
    sticks{2} = [lkne; lank];
    sticks{3} = [rhip; rkne];
    sticks{4} = [rkne; rank];    
else
    assert(false);
end


for i = 1:numel(sticks)
    plot(sticks{i}(:,1)', sticks{i}(:,2)', style);
end

end
