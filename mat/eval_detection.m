function testdata = eval_detection(testdata)

%% get gt-box
for i=1:numel(testdata)
    joints = testdata(i).point(1:14,:);    % use 14 joints
    % get rid of occluded joints
    occ = testdata(i).occ;
    occ = occ(1:14);
    joints(find(occ==1),:) = [];
    
    x_min = min(joints(:,1));
    y_min = min(joints(:,2));
    x_max = max(joints(:,1));
    y_max = max(joints(:,2));
    testdata(i).gtbox = [x_min y_min x_max-x_min y_max-y_min];
end

%% compute overlapping area and overlapping score
IoU = [];
for i=1:numel(testdata)
    
    % get intersection
    inter.x1 = max(testdata(i).box(1), testdata(i).gtbox(1));
    inter.x2 = min(testdata(i).box(1)+testdata(i).box(3), testdata(i).gtbox(1)+testdata(i).gtbox(3));
    inter.y1 = max(testdata(i).box(2), testdata(i).gtbox(2));
    inter.y2 = min(testdata(i).box(2)+testdata(i).box(4), testdata(i).gtbox(2)+testdata(i).gtbox(4));

    % compute area of intersection
    if inter.x1 < inter.x2 && inter.y1 < inter.y2   % intersection exists
        inter.area = (inter.x2-inter.x1)*(inter.y2-inter.y1);
    else
        inter.area = 0;
    end
    
    % compute area of gt box and prediction box
    area_gt = testdata(i).gtbox(3) * testdata(i).gtbox(4);
    area_pred = testdata(i).box(3) * testdata(i).box(4);
    
    % overlapping score: [intersection over union overlap]
    IoU(i) = inter.area / (area_gt+area_pred-inter.area);
    assert(IoU(i) >= 0);
end

% fprintf('[detection] mean IoU: %.4f \n', mean(IoU));
fprintf('%.4f \n', mean(IoU));

end