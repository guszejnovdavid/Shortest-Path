function short_path_cost()
% Finds the shortest path in a Cost using reindforcement learning
% Monte-Carlo

tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Parameters

Param.ngrid=15; %grid size
Param.gamma=1.0; %discounting parameter, exp. reward= sum(gamma^i*reward_i)
Param.sightrange=5*Param.ngrid; %maximum number of steps used in calculating expected rewrad
Param.alpha=0.1; %policy update step size: updatedpolicy=oldpolicy+alpha*(newpolicy-oldpolicy)
Param.beta=0.1; %reward update step size: updated expected reward=oldreward+beta*(newreward-oldreward)
Param.stepcost=-1; %cost for taking a step
Param.outofboundpenalty=0.0; %penalty for going out of bounds
Param.endreward=0.0; %reward for getting to the en
Param.maxstepnum=Param.ngrid^2; %Maximum number of steps for random walk
Param.nsample=Param.ngrid^2; %sample size for each policy
Param.nepoch=2000; %number of epochs (number of policy updates)
Param.startX=1;Param.startY=1; %starting position
Param.endX=Param.ngrid-1;Param.endY=Param.ngrid-1; %ending position

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Shorthands

npoints=Param.ngrid^2;
%grid coordinates
Ymatrix=kron(ones(Param.ngrid,1),1:Param.ngrid);
Xmatrix=Ymatrix';
%Flatten
gridY=Ymatrix(:)';
gridX=Xmatrix(:)';
%Starting and ending point
startpoint=Param.startX+(Param.startY-1)*Param.ngrid;
endpoint=Param.endX+(Param.endY-1)*Param.ngrid;
%summation coefficients (for discounting)
sumcoeffs=Param.gamma.^(0:Param.sightrange);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Cost made up from 2 Gaussian

gaussfunction=@(z,z0,sigma) exp(-(z-z0).^2./(2*sigma^2));
%Gaussian 1
Cost.Amplitude1=-3;
Cost.x1=1;Cost.y1=Param.ngrid;
Cost.sigma1=round(Param.ngrid*0.3);
%Gaussian 2
Cost.Amplitude2=-3;
Cost.x2=round(Param.ngrid*0.5);Cost.y2=round(Param.ngrid*0.5);
Cost.sigma2=round(Param.ngrid*0.25);
%Cost function
Cost.Costfunction=@(x,y) Cost.Amplitude1*gaussfunction(x,Cost.x1,Cost.sigma1).*gaussfunction(y,Cost.y1,Cost.sigma1)+...
    Cost.Amplitude2*gaussfunction(x,Cost.x2,Cost.sigma2).*gaussfunction(y,Cost.y2,Cost.sigma2);
%Evaluate Cost function
Cost.values=Cost.Costfunction(gridX,gridY);
%Add endpoint reward
Cost.values(endpoint)=Cost.values(endpoint)+Param.endreward;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initialize policy

policy=ones(npoints,4).*0.25; %for each point equal probability of going in either 4 directions
%Stored results
expectedreward=zeros(npoints,4); %setting it zero ensures that we will explore the parameter space


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Start learning

for iepoch=1:Param.nepoch %for each epoch
    if ~mod(iepoch,100)
        disp([num2str(iepoch) '/' num2str(Param.nepoch) ' done']);
    end
    %Results from MC realization at this epoch
    MCResults.position=zeros(Param.nsample,Param.maxstepnum); %point visited in each sample
    MCResults.position(:,1)=startpoint; %initial point
    MCResults.choice=zeros(Param.nsample,Param.maxstepnum); %choice at each step
    MCResults.reward=zeros(Param.nsample,Param.maxstepnum+Param.sightrange); %reward gained for this specific choice, filling ends with zeros
    MCResults.longreward=zeros(Param.nsample,Param.maxstepnum); %long term reward gained for taking this choice )discounted)
    for jsample=1:Param.nsample %for each MC realization
        %Start random walk
        for k=1:(Param.maxstepnum-1)
            %choose direction based on policy and position
            MCResults.choice(jsample,k)=make_choice(policy,MCResults.position(jsample,k));
            %move into that direction
            [MCResults.position(jsample,k+1),MCResults.reward(jsample,k)]=make_move(MCResults.position(jsample,k),MCResults.choice(jsample,k),Cost.values,Param);
            %check if we arrived at our destination
            if (MCResults.position(jsample,k+1)==endpoint)
                break; %break loop, rest of the rewards are zero
            end
        end
        %Calculate long term reward
        for k=1:(Param.maxstepnum-1)
            MCResults.longreward(jsample,k)=sum( sumcoeffs.*MCResults.reward(jsample,k:(k+Param.sightrange)) );
        end
    end %end of MC ralization
    
    %Update rewards for different actions
    for jpoint=1:npoints %for eachpoint
        posindex=(MCResults.position(:)==jpoint); %each time we get to this point
        if sum(posindex)
            for k=1:4 %for each action
                ind=posindex &(MCResults.choice(:)==k);%find right indices
                %update expected reward
                if sum(ind)
                    expectedreward(jpoint,k)=expectedreward(jpoint,k)+Param.beta.*(mean(MCResults.longreward(ind))-expectedreward(jpoint,k));
                end
            end
            %get transition probabilities based on just this epoch
            probs=exp(expectedreward(jpoint,:))./sum(exp(expectedreward(jpoint,:)));
            %update policy
            policy(jpoint,:)=policy(jpoint,:)+Param.alpha*(probs-policy(jpoint,:));
        end
    end
    
end % end of epochs


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Output

Valuefunction=sum(expectedreward.*policy,2);
Valuematrix=reshape(Valuefunction,Param.ngrid,Param.ngrid);

figure(1),
[C,h]=contourf(Xmatrix,Ymatrix,Valuematrix);
clabel(C,h);
hold on
%overplot Cost
contour(Xmatrix,Ymatrix,reshape(Cost.values,Param.ngrid,Param.ngrid))
hold off
title('Value function')
xlabel('x')
ylabel('y')

% %Plot gradient
% figure(2)
% [dValueX,dValueY]=gradient(Valuematrix);%gradient
% %plot Cost
% contour(Xmatrix,Ymatrix,reshape(Cost.values,Param.ngrid,Param.ngrid))
% hold on
% %gradient arrows
% quiver(Xmatrix,Ymatrix,dValueX,dValueY);
% hold off
% title('Value function gradient')
% xlabel('x')
% ylabel('y')

%Plot most likely direction
figure(3)
arrowX=reshape(policy(:,1)-policy(:,2),Param.ngrid,Param.ngrid);
arrowY=reshape(policy(:,3)-policy(:,4),Param.ngrid,Param.ngrid);
%plot Cost
contour(Xmatrix,Ymatrix,reshape(Cost.values,Param.ngrid,Param.ngrid))
hold on
%gradient arrows
quiver(Xmatrix,Ymatrix,arrowX,arrowY);
hold off
title('Most likely direction of motion')
xlabel('x')
ylabel('y')

%Saving figure
savefilename=['Cost_path_'];
saveas(1,[savefilename '_value_function.eps'],'epsc');
%saveas(2,[savefilename '_value_function_gradient.eps'],'epsc');
saveas(3,[savefilename '_most_likely_path.eps'],'epsc');

toc;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to make choice based on policy

function choice=make_choice(policy,position)
CDF=cumsum(policy(position,:));
choice=find(CDF>rand(),1,'first'); %find which label we choose
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to calculate new position and the reward for the move

function [newpos,reward]=make_move(pos,choice,Cost,Param)

reward=Param.stepcost;
x=mod(pos,Param.ngrid);
if x==0
    x=x+Param.ngrid;
end
y=(pos-x)/Param.ngrid+1;
newx=x;newy=y;
switch choice %make move
    case 1
        newx=x+1;
    case 2
        newx=x-1;
    case 3
        newy=y+1;
    case 4
        newy=y-1;
end

%check for boundaries
if ( (newy<1) || (newy>Param.ngrid) || (newx<1) || (newx>Param.ngrid) )
    newpos=pos; %don't move
    reward=reward+Param.outofboundpenalty;
else
    newpos=newx+(newy-1)*Param.ngrid; %new position
    %get reward for step
    reward=reward+Cost(newpos);
end

end
