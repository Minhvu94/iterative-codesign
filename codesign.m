%% Steer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
load('dis_info.mat')

dt = 0.01;
T = 2;
iter_max = ceil(T/dt);

x0 = [0;pi;0;0];
x_target = [0;0;0;0];

% u = 0.1*ones(iter_max,1);
u = 0.3*rand(iter_max,1);

net = []; NN_input = []; NN_output = [];

rep = 1;
count = 0;
rejected = false;
lambda_coeff = 0.01;
store_data = [];
continue_iterating = true;
while continue_iterating  % Iterating scheme: U --> trajactory [x(0)...x(N)] --> H --> U' 
    if rejected==false
        x = x0;
        x_traj = x0;
        % Apply u to the real system
        for iter = 1:iter_max                    
            [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u(iter)]); 
            x = x_trajJ_fine(end,:)'; 
            x = x(1:4);
            x_traj = [x_traj, x];      
        end
        xn = x; 
        
        % train NN 
        [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj,u); 
        [net,x1_step1,y1_step1,NN_input,NN_output] = sampling_and_train(x0,net,NN_input,NN_output);
        
        for iter = 1:iter_max  
            % Prepare A,B to later calculate H        
            R_big = JPhi(dt,x_traj(1,iter),x_traj(2,iter),x_traj(3,iter),x_traj(4,iter),u(iter));
            A_real_store{iter} = R_big(1:4,1:4);     % A in Ax+Bu
            B_real_store{iter} = R_big(1:4,5:5);     % B in Ax+Bu

            jac = NN_jacob(net, [x_traj(:,iter);u(iter)], x1_step1, y1_step1);
            A_store{iter} = jac(:,1:4);
            B_store{iter} = jac(:,5);
        end
        
        cost_old = norm(xn-x_target);       
        if cost_old <= 0.01
            cost_old
            break
        end

        % Calculate H of NN-model and the H_real of the system:
        H_real = B_real_store{1};
        for iter = 2:iter_max
            H_real = [A_real_store{iter}*H_real, B_real_store{iter}];
        end

        H = B_store{1};
        for iter = 2:iter_max
            H = [A_store{iter}*H, B_store{iter}];  
        end
        
        RMSE = sqrt(mean((H(:)-H_real(:)).^2));
        
        figure(1)
        clf
        hold on 
        plot3(NN_input(2,:),NN_input(4,:),NN_input(5,:),'.');
        plot3(x_traj(2,1:end-1),x_traj(4,1:end-1),u','r','LineWidth',2)
        view(2);
        xlabel('theta 2')
        ylabel('theta 2 dot')
        zlabel('u')
        axis([ -2 5 -15 10 -10 10]);
        drawnow;
        
    end    
    
    % Solve u_proposal
    lambda = lambda_coeff*cost_old^1.5;
    du_proposal = -(H'*H+lambda*eye(iter_max))\(H'*(xn-x_target));
    u_proposal = u + du_proposal;
    
    % Simulate real system using proposed u
    x_sim = x0;
    x_traj_sim = x0;
    for iter = 1:iter_max                      
        [~, x_trajJ_fine_sim] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x_sim;u_proposal(iter)]); 
        x_sim = x_trajJ_fine_sim(end,:)'; 
        x_sim = x_sim(1:4);
        x_traj_sim = [x_traj_sim, x_sim]; 
    end
%     [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj_sim,u_proposal);
    
    cost_proposal_actual = norm(x_sim-x_target);
    
    % See if the approx. of H slows down the progress 
    if cost_old - cost_proposal_actual < 0.01
        count = count + 1;
    else
        count = 0;
    end
    
    if cost_proposal_actual < cost_old      
        lambda_coeff = 0.98*lambda_coeff;
        u = u_proposal;
        rejected = false;     % accept u_proposal due to cost benefits
    else                                    
        lambda_coeff = 1.2*lambda_coeff;
        rejected = true;      % reject u_proposal
    end
    
    store_data = [store_data [cost_old;RMSE]];
    rep = rep + 1;

    fprintf('rep=%.0f; cost = %.4f; lamda = %.4f; reject=%.0f; slow=%.0f; RMSE = %.4f; use last %.0f trajectories;\n',[rep, cost_old, lambda_coeff, rejected, count, RMSE, size(NN_input,2)/(size(x_traj,2)-1)]);

end
dt*norm(u)^2
step1_u = u;
disp( 'Done1!' );
Minh_Animation(x_traj)

%% Minh - Animation:

function Minh_Animation(x_traj)
points = [0  0  0;
          0 -8 -8;     
          0  0 -12];
   
figure
clf
hold on 
grid on 
view([340 25])
plot_link1 = plot3(points(1,1:2),points(2,1:2),points(3,1:2),'b','LineWidth',5.5);
plot_link2 = plot3(points(1,2:3),points(2,2:3),points(3,2:3),'k','LineWidth',4);

axis([ -13 13 -13 13 -13 13]);
plot3([0,0],[0,0],[0,-13],'r','LineWidth',7);

for ii=1:length(x_traj)

Rz = makehgtform('zrotate',x_traj(1,ii));
Rz_points = Rz(1:3,1:3)*points;

M = makehgtform('axisrotate',Rz_points(:,2)',-(x_traj(2,ii)-pi));
R_link1 = M(1:3,1:3)*Rz_points;

delete(plot_link1)
delete(plot_link2)

plot_link1 = plot3(Rz_points(1,1:2),Rz_points(2,1:2),Rz_points(3,1:2),'b','LineWidth',5);
plot_link2 = plot3(R_link1(1,2:3),R_link1(2,2:3),R_link1(3,2:3),'k','LineWidth',3);

drawnow;
pause(0.01)
end

rotate3d on

end


function [net,x1_step1,y1_step1,NN_input,NN_output] = sampling_and_train(x_start_sample,net,NN_input,NN_output)

% -------------------- training --------------------
if isempty(net)
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
    hiddenLayerSize = [20 20];
    net = fitnet(hiddenLayerSize,trainFcn);

    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 5/100;

    net.trainParam.epochs=250;
    net.performFcn = 'mse';  % Mean Squared Error

    [net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','yes');
else
    net.trainParam.epochs=250;
    % Train the Network
    [net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','no');
    % Retrain if needed
    retrain = 0;
    while strcmp(tr.stop,'Validation stop.') && tr.num_epochs<100 && retrain<3
    %     display('retrain due to bad training!')
        net.trainParam.epochs=400;
        [net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','no');
        retrain = retrain + 1;
    end
end

% Save the NN 
genFunction(net,'myNN','ShowLinks','no');

% Automatically replace [Y,Xf,Af] with [Y,Xf,Af,x1_step1,y1_step1]
A = regexp( fileread('myNN.m'), '\n', 'split');
A{1} = sprintf('function [Y,Xf,Af,x1_step1,y1_step1] = %s(X,~,~)','myNN');
fid = fopen('myNN.m', 'w');
fprintf(fid, '%s\n', A{:});
fclose('all');

[~,~,~,x1_step1,y1_step1] = myNN(rand(5,1));
end

function [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj,u)
    % Using 4 last trajectories, removing the rest 
    k = 4;
    l_traj = size(x_traj,2)-1;
    if size(NN_input,2) > l_traj*k        
        NN_input = NN_input(:,end-l_traj*k+1:end);
        NN_output = NN_output(:,end-l_traj*k+1:end);
    end
    
    % Add the current trajectory 
    NN_input = [NN_input, [x_traj(:,1:end-1);u'] ];
    NN_output = [NN_output, x_traj(:,2:end)];
end

