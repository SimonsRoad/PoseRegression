function [pcp1,pcp2,pcp3] = pcp_eval(ca,gt,thresh,mode)
% the function is used to calculate the pcp score of body with and without
% occlude part
% ca    : struct which contains 'point' field with test 14 points
% gt    : struct which contains 'point' field with ground truth 14 points
%         ;'state' field which contains the occlusion state of that part
%         1 for visible; 2 for self-occlude 3 for other-occlude; 'quality'
%         field which contains the quality of the instance, 1 for good, 0 for bad;
% thresh: scale the threshold
% mode  : 'u' for only calculate upper body(8 points),'l' for only calculate
%          lower body,'a' for calculate the whole body
% pcp1  : score including both other-occlude parts and self-occlude parts
% pcp2  : score including self-occlude and excluding other-occlude
% pcp3  : score excluding both self-occlude and other-occlude

if nargin < 4
    thresh  = 0.5;
    mode='a';
end

if nargin<5
    mode='a';
end

assert(numel(ca) == numel(gt));
point_num=size(gt(1).point,1);

% end points of the stick
if point_num==14
    stick=[1,2,4,6,2,3,5,10,12,9,11,4,3;2,4,6,8,3,5,7,12,14,11,13,10,9];
elseif point_num==10
    stick=[1,2,4,6,2,3,5,4,3;2,4,6,8,3,5,7,10,9];
end

stick_num=size(stick,2);
% L is length of stick in ground truth
L=zeros(stick_num,1);
% stick_dist is length of stick in test result
stick_dist=zeros(stick_num,1);
dist=zeros(point_num,1);

state1=zeros(stick_num,1); % state num 1 for visiable and self-occlude 
state2=zeros(stick_num,1); % state num 1 for visiable
j=1;
for i=1:length(gt)
    if gt(i).quality
        dist(:,j)=sqrt(sum((ca(i).point-gt(i).point).^2,2));
        temp=dist(:,j);
        stick_dist(:,j)=max(temp(stick));
        for k=1:stick_num
            L(k,j)=sqrt(sum((gt(i).point(stick(1,k),:)-gt(i).point(stick(2,k),:)).^2,2));
            state1(k,j)=(gt(i).state(stick(1,k))-3)&(gt(i).state(stick(2,k))-3);
            state2(k,j)=~((gt(i).state(stick(1,k))-1)|(gt(i).state(stick(2,k))-1));
        end
        j=j+1;
    end
end

matrix=stick_dist<=thresh*L;

if mode=='u'
    part_pcp=mean(matrix(1:7,:),2);
    pcp1=mean(part_pcp*100);
    pcp2=mean((sum(matrix(1:7,:).*state1(1:7,:),2)./sum(state1(1:7,:),2))*100);
    pcp3=mean((sum(matrix(1:7,:).*state2(1:7,:),2)./sum(state2(1:7,:),2))*100);  
elseif mode=='l'
    part_pcp=mean(matrix(8:11,:),2);
    pcp1=mean(part_pcp*100);
    pcp2=mean((sum(matrix(8:11,:).*state1(8:11,:),2)./sum(state1(8:11,:),2))*100);
    pcp3=mean((sum(matrix(8:11,:).*state2(8:11,:),2)./sum(state2(8:11,:),2))*100);
else
    part_pcp=mean(matrix,2);
    pcp1=mean(part_pcp*100);
    pcp2=mean((sum(matrix.*state1,2)./sum(state1,2))*100);
    pcp3=mean((sum(matrix.*state2,2)./sum(state2,2))*100);
end

