clear
clc
close all


syms xi1 xi2 xi3 xi4 xi5 dt


% furuta
res = furuta('model');
F= [xi3;
    xi4;
    res(1)
    res(2)
    0];
F = simplify(F);

p = 4;

% [Phi,Psi_p,JPhi] = compute_Phi_and_JPhi(p,F,[xi1 xi2 xi3 xi4 xi5],dt);
[Phi_real,Psi_p_real,JPhi_real] = compute_Phi_and_JPhi(p,F,[xi1 xi2 xi3 xi4 xi5],dt);


disp( 'Done!' );


% save('cart_pole_model.mat','Phi','Psi_p','JPhi')
save('furuta_real.mat','Phi_real','Psi_p_real','JPhi_real')


%% Part 1: Sample --> train NN --> steer until slow
clear
clc
close all
load('dis_info.mat')


% Sample from x_start_sample
x_start_sample = [0 pi 0 0]';

dt = 0.01;
T = 2;
iter_max = ceil(T/dt);

tt = dt:dt:T;

signal_store = [ones(iter_max,1), sin(2*tt)', cos(2*tt)', sin(4*tt)', cos(4*tt)', sin(6*tt)', cos(6*tt)', sin(8*tt)', cos(8*tt)', sin(10*tt)', cos(10*tt)',sin(12*tt)', cos(12*tt)'];
nbasis = size(signal_store,2);


NN_input = [];
NN_output = [];



for ittt = 0:8
    ittt
    
    u_test = [];
    
    alpha = (1+(mod(ittt,9)))*(rand(nbasis,1)-.5)*2;
    
    alpha = 1*alpha;
    
    u = signal_store*alpha;
    
    u_test = u;
    
    for perturb_times = 1:1
        
        x0 = x_start_sample.*((rand(size(x_start_sample))-.5)*.3 +1);
        x = x0;
        x_traj = [x0];
        
        for iter = 1:size(u_test,1)
            
            % real system
            [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u_test(iter)]);
            x = x_trajJ_fine(end,:)';
            x = x(1:4);
            x_traj = [x_traj x];
            
        end
        
        x_traj(2,:) = sign(x_traj(2,:)).*mod(abs(x_traj(2,:)),2*pi);
        
        NN_input = [NN_input, [x_traj(2:4,1:end-1);u_test'], -[x_traj(2:4,1:end-1);u_test']];
        NN_output = [NN_output, [x_traj(3,2:end), -x_traj(3,2:end);
            [x_traj(4,2:end), -x_traj(4,2:end)]]];
        
    end
    
    
    FI = find((-pi <=  NN_input(1,:)));
    NN_input = NN_input(:,FI);
    NN_output = NN_output(:,FI);
    
    FI = find(abs(NN_input(4,:))<=10);
    NN_input = NN_input(:,FI);
    NN_output = NN_output(:,FI);
    
    clf
    plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
    view(2);
    drawnow;
end


rotate3d on

%% Part 1: Sample --> train NN --> steer until slow %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
load('dis_info.mat')


% Sample from x_start_sample
x_start_sample = [0 pi 0 0]';

dt = 0.01;
T = 2;
iter_max = ceil(T/dt);

tt = dt:dt:T;

signal_store = [ones(iter_max,1), sin(2*tt)', cos(2*tt)', sin(4*tt)', cos(4*tt)', sin(6*tt)', cos(6*tt)', sin(8*tt)', cos(8*tt)', sin(10*tt)', cos(10*tt)',sin(12*tt)', cos(12*tt)'];
nbasis = size(signal_store,2);


NN_input = [];
NN_output = [];


for k_sample = 1:5
                
        alpha = 0.3*k_sample*(rand(nbasis,1)-.5)*2;              
        u_test = signal_store*alpha;        
%         [max(u_test), min(u_test)]       
                   
        x0 = x_start_sample + 0.1*(rand(length(x_start_sample),1)-.5)*2;
        x = x0;
        x_traj = x0;
        
        % real system
        for iter = 1:size(u_test,1)            
            [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u_test(iter)]);
            x = x_trajJ_fine(end,:)';
            x = x(1:4);
            x_traj = [x_traj x];
        end

%         x_traj(1,:) = sign(x_traj(1,:)).*mod(abs(x_traj(1,:)),2*pi);
        x_traj(2,:) = sign(x_traj(2,:)).*mod(abs(x_traj(2,:)),2*pi);

        NN_input = [NN_input, [x_traj(2:4,1:end-1);u_test'] ];
        NN_output = [NN_output, x_traj(3:4,2:end)];
%         NN_input = [NN_input, [x_traj(2:4,1:end-1);u_test'], -[x_traj(2:4,1:end-1);u_test']];
%         NN_output = [NN_output, [x_traj(3,2:end), -x_traj(3,2:end);
%                                  x_traj(4,2:end), -x_traj(4,2:end)] ];

          
        
        FI = find((-pi <=  NN_input(1,:)));
        NN_input = NN_input(:,FI);
        NN_output = NN_output(:,FI);
        
        FI = find(abs(NN_input(4,:))<=10);
        NN_input = NN_input(:,FI);
        NN_output = NN_output(:,FI);
        
        clf
        plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
        view(3);
        xlabel('theta 2')
        ylabel('theta 2 dot')
        zlabel('u')
        axis([ -10 10 -30 30 -20 20]);

        drawnow;
end
    
rotate3d on

%% train NN
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = [20 20];
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 5/100;

net.trainParam.epochs=400;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Train the Network
[net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','yes');
% view(net)
genFunction(net,'myNN');
% [Y,Xf,Af,x1_step1,y1_step1]


% automatically replace [Y,Xf,Af] with [Y,Xf,Af,x1_step1,y1_step1]
A = regexp( fileread('myNN.m'), '\n', 'split');
A{1} = sprintf('function [Y,Xf,Af,x1_step1,y1_step1] = %s(X,~,~)','myNN');
fid = fopen('myNN.m', 'w');
fprintf(fid, '%s\n', A{:});
fclose('all');

[~,~,~,x1_step1,y1_step1] = myNN(rand(4,1));
NN_input_prev = NN_input;
NN_output_prev = NN_output;

%% test how well NN can capture the dynamics 
x0 = [0;pi;0;0];
iter_max = 50;
u = 0.1*ones(iter_max,1);
x = x0;
x_traj = [];
[~,~,~,x1_step1,y1_step1] = myNN(rand(4,1));

for iter = 1:iter_max
    % Prepare A,B to later calculate H
    R_big = JPhi(dt,x(1),x(2),x(3),x(4),u(iter));
    A_real_store{iter} = R_big(1:4,1:4);     % A in Ax+Bu
    B_real_store{iter} = R_big(1:4,5:5);     % B in Ax+Bu
    
    jac = NN_jacob(net, [x(2:4);u(iter)], x1_step1, y1_step1);
    A_store{iter} = [1    0          dt          0;
                     0    1          0           dt;
                     0 jac(1,1)  jac(1,2)  jac(1,3);
                     0 jac(2,1)  jac(2,2)  jac(2,3);];
    B_store{iter} = [0;
                     0;
                     jac(1,4);
                     jac(2,4)];
    
    % The actual trajectory using adaptive step size mechanism
    [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u(iter)]);
    x = x_trajJ_fine(end,:)';
    x = x(1:4);
    x_traj = [x_traj, x];
end

% Calculate H:
H = B_store{1};
for iter = 2:iter_max
    H = [A_store{iter}*H, B_store{iter}];
end
    
H_real = B_real_store{1};
for iter = 2:iter_max
    H_real = [A_real_store{iter}*H_real, B_real_store{iter}];
end

RMSE = sqrt(mean((H(:)-H_real(:)).^2))  

%% Steer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
dt = 0.01;
T = 2;
iter_max = ceil(T/dt);

x0 = [0;pi;0;0];
x_target = [0;0;0;0];
% x_target = [0.25;pi-0.2;0;0];

u = 0.1*ones(iter_max,1);
% u = 0.1*rand(iter_max,1);
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
        [net,x1_step1,y1_step1,NN_input,NN_output] = sampling_and_train(x_start_sample,net,NN_input,NN_output);
        
        for iter = 1:iter_max  
            % Prepare A,B to later calculate H        
            R_big = JPhi(dt,x_traj(1,iter),x_traj(2,iter),x_traj(3,iter),x_traj(4,iter),u(iter));
            A_real_store{iter} = R_big(1:4,1:4);     % A in Ax+Bu
            B_real_store{iter} = R_big(1:4,5:5);     % B in Ax+Bu

            jac = NN_jacob(net, [x_traj(2:4,iter);u(iter)], x1_step1, y1_step1);
            A_store{iter} = [1    0          dt          0;
                             0    1          0           dt;
                             0 jac(1,1)  jac(1,2)  jac(1,3);
                             0 jac(2,1)  jac(2,2)  jac(2,3);];
            B_store{iter} = [0;
                             0;
                             jac(1,4);
                             jac(2,4)];
        end
        
        cost_old = norm(xn-x_target);       
        if cost_old <= 0.01
            cost_old
            break
        end

        % Calculate H:
        [H, H_real, RMSE] = calculate_H_RMS(iter_max,A_store,B_store,A_real_store,B_real_store);
%         H_real = B_real_store{1};
%         for iter = 2:iter_max
%             H_real = [A_real_store{iter}*H_real, B_real_store{iter}];
%         end
% 
%         H = B_store{1};
%         for iter = 2:iter_max
%             H = [A_store{iter}*H, B_store{iter}];  
%         end
% 
%         RMSE = sqrt(mean((H(:)-H_real(:)).^2));
        
        figure(1)
        clf
        hold on 
        plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
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
%         % Prepare A,B to later calculate H
%             R_big = JPhi(dt,x_sim(1),x_sim(2),x_sim(3),x_sim(4),u_proposal(iter));
%             A_real_store_sim{iter} = R_big(1:4,1:4);     % A in Ax+Bu
%             B_real_store_sim{iter} = R_big(1:4,5:5);     % B in Ax+Bu
% 
%             jac = NN_jacob(net, [x_sim(2:4);u_proposal(iter)], x1_step1, y1_step1);
%             A_store_sim{iter} = [1    0          dt          0;
%                              0    1          0           dt;
%                              0 jac(1,1)  jac(1,2)  jac(1,3);
%                              0 jac(2,1)  jac(2,2)  jac(2,3);];
%             B_store_sim{iter} = [0;
%                              0;
%                              jac(1,4);
%                              jac(2,4)];

                         
        [~, x_trajJ_fine_sim] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x_sim;u_proposal(iter)]); 
        x_sim = x_trajJ_fine_sim(end,:)'; 
        x_sim = x_sim(1:4);
        x_traj_sim = [x_traj_sim, x_sim]; 
    end
%     [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj_sim,u_proposal);
    
    cost_proposal_actual = norm(x_sim-x_target);
    
    if cost_old - cost_proposal_actual < 0.01
%         [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj_sim,u_proposal);
%         RMSE = calculate_H_RMS_sim(iter_max,A_store_sim,B_store_sim,A_real_store_sim,B_real_store_sim);
        count = count + 1;
%         if count >= 5
% %             Minh_Animation(x_traj)
%             display('Start sampling and train!')
%             pause()
%             [net,x1_step1,y1_step1,NN_input_prev,NN_output_prev,NN_input,NN_output] = sampling_and_train(xn,net,NN_input,NN_output);
%             NN_input = [];
%             NN_output = [];
%             count = 0;
%         end
    else
%         NN_input = [];
%         NN_output = [];
        count = 0;
    end
    
    if cost_proposal_actual < cost_old      
        lambda_coeff = 0.95*lambda_coeff;
        u = u_proposal;
        rejected = false;     % accept u_proposal due to cost benefits
    else                                    
        lambda_coeff = 1.2*lambda_coeff;
        rejected = true;      % reject u_proposal
    end
    
%     display('here2')
    store_data = [store_data [cost_old;RMSE]];
    rep = rep + 1;

    fprintf('rep=%.0f; cost = %.4f; lamda = %.4f; reject=%.0f; count=%.0f; RMSE = %.4f\n',[rep, cost_old, lambda_coeff, rejected, count, RMSE]);
% avv

end
% x_start_sample = xn;
% dt*norm(u)^2
% step1_u = u;
% disp( 'Done1!' );

%%
% Minh_Animation(x_traj)
figure
% clf
% hold on
% plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
% mmm = plot3(x_traj(2,2:end),x_traj(4,2:end),u,'r');
% mmm2 = scatter3(x_start_sample(2),x_start_sample(4),0,100,'r')
plot(store_data(1,:))

%%
            [net,x1_step1,y1_step1] = sampling_and_train(xn,net,NN_input,NN_output);

%% Minh - Animation:
clc
for i=5:5
    i
    x_traj = traj_1(i*4-3:i*4,:);
    Minh_Animation(x_traj)
end


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


function [net,x1_step1,y1_step1] = sampling_and_train1(x_start_sample,net,NN_input,NN_output)
load('dis_info.mat')

dt = 0.01;
n_of_random_steps = 10;

tt = dt:dt:dt*n_of_random_steps;

signal_store = [ones(n_of_random_steps,1), sin(2*tt)', cos(2*tt)', sin(4*tt)', cos(4*tt)', sin(6*tt)', cos(6*tt)', sin(8*tt)', cos(8*tt)', sin(10*tt)', cos(10*tt)',sin(12*tt)', cos(12*tt)'];
nbasis = size(signal_store,2);

% NN_input = [];
% NN_output = [];

for ittt = 0:8
    ittt
    
    u_test = [];
    
    alpha = (1+(mod(ittt,9)))*(rand(nbasis,1)-.5)*2;
    
    alpha = 1*alpha;
    
    u = signal_store*alpha;
    
    u_test = u;
    
    for perturb_times = 1:1
        
        x0 = x_start_sample.*((rand(size(x_start_sample))-.5)*.3 +1);
        x = x0;
        x_traj = [x0];
        
        for iter = 1:size(u_test,1)
            
            % real system
            [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u_test(iter)]);
            x = x_trajJ_fine(end,:)';
            x = x(1:4);
            x_traj = [x_traj x];
            
        end
        
        x_traj(2,:) = sign(x_traj(2,:)).*mod(abs(x_traj(2,:)),2*pi);
        
        NN_input = [NN_input, [x_traj(2:4,1:end-1);u_test'], -[x_traj(2:4,1:end-1);u_test']];
        NN_output = [NN_output, [x_traj(3,2:end), -x_traj(3,2:end);
            [x_traj(4,2:end), -x_traj(4,2:end)]]];
        
    end
    
    
    FI = find((-pi <=  NN_input(1,:)));
    NN_input = NN_input(:,FI);
    NN_output = NN_output(:,FI);
    
    FI = find(abs(NN_input(4,:))<=10);
    NN_input = NN_input(:,FI);
    NN_output = NN_output(:,FI);
end

figure(1)
clf
hold on 
plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
scatter3(x_start_sample(2),x_start_sample(4),0,100,'r')
view(3);
xlabel('theta 2')
ylabel('theta 2 dot')
zlabel('u')
drawnow;

pause()

net.trainParam.epochs=400;

% Train the Network
[net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','yes');
genFunction(net,'myNN');

% automatically replace [Y,Xf,Af] with [Y,Xf,Af,x1_step1,y1_step1]
A = regexp( fileread('myNN.m'), '\n', 'split');
A{1} = sprintf('function [Y,Xf,Af,x1_step1,y1_step1] = %s(X,~,~)','myNN');
fid = fopen('myNN.m', 'w');
fprintf(fid, '%s\n', A{:});
fclose('all');

[~,~,~,x1_step1,y1_step1] = myNN(rand(4,1));

end


function [net,x1_step1,y1_step1,NN_input,NN_output] = sampling_and_train(x_start_sample,net,NN_input,NN_output)
load('dis_info.mat')

dt = 0.01;
n_of_samples = 0;
n_of_random_steps = 10;

tt = dt:dt:dt*n_of_random_steps;

signal_store = [ones(n_of_random_steps,1), sin(2*tt)', cos(2*tt)', sin(4*tt)', cos(4*tt)', sin(6*tt)', cos(6*tt)', sin(8*tt)', cos(8*tt)', sin(10*tt)', cos(10*tt)',sin(12*tt)', cos(12*tt)'];
nbasis = size(signal_store,2);


for k_sample = 1:n_of_samples
        
        alpha = 0.5*k_sample*(rand(nbasis,1)-.5)*2;        
        u_test = signal_store*alpha;
        
        x = x_start_sample;
        x_traj = x_start_sample;

        % Apply random u_test to real system 
        for iter = 1:n_of_random_steps
            [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u_test(iter)]);
            x = x_trajJ_fine(end,:)';
            x = x(1:4);
            x_traj = [x_traj x];
        end

        x_traj(2,:) = sign(x_traj(2,:)).*mod(abs(x_traj(2,:)),2*pi);

        NN_input = [NN_input, [x_traj(2:4,1:end-1);u_test'] ];
        NN_output = [NN_output, [x_traj(3,2:end);x_traj(4,2:end)] ];                  
        
        % remove data that are too large (outside the neccessary area)                       
        FI = find((-pi <=  NN_input(1,:)));
        NN_input = NN_input(:,FI);
        NN_output = NN_output(:,FI);
        
        FI = find(abs(NN_input(4,:))<=10);
        NN_input = NN_input(:,FI);
        NN_output = NN_output(:,FI);
end

% figure(1)
% clf
% hold on 
% plot3(NN_input(1,:),NN_input(3,:),NN_input(4,:),'.');
% scatter3(x_start_sample(2),x_start_sample(4),0,100,'r')
% view(3);
% xlabel('theta 2')
% ylabel('theta 2 dot')
% zlabel('u')
% drawnow;
% 
% pause()

net.trainParam.epochs=250;

% Train the Network
[net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','no');

if strcmp(tr.stop,'Validation stop.') && tr.num_epochs<100
    display('retrain due to bad training!')
    net.trainParam.epochs=400;
    [net,tr] = train(net,NN_input,NN_output,'useParallel','yes','showResources','no');
end
genFunction(net,'myNN','ShowLinks','no');

% automatically replace [Y,Xf,Af] with [Y,Xf,Af,x1_step1,y1_step1]
A = regexp( fileread('myNN.m'), '\n', 'split');
A{1} = sprintf('function [Y,Xf,Af,x1_step1,y1_step1] = %s(X,~,~)','myNN');
fid = fopen('myNN.m', 'w');
fprintf(fid, '%s\n', A{:});
fclose('all');

[~,~,~,x1_step1,y1_step1] = myNN(rand(4,1));
% NN_input_prev = NN_input;
% NN_output_prev = NN_output;
end

function [NN_input,NN_output] = store_training_data(NN_input,NN_output,x_traj,u)
    % Remove the furthest past trajectory if the data is getting too long 
    if size(NN_input,2) > 2000      
        NN_input = NN_input(:,size(x_traj,2):end);
        NN_output = NN_output(:,size(x_traj,2):end);
    end
    x_traj(2,:) = sign(x_traj(2,:)).*mod(abs(x_traj(2,:)),2*pi);
    NN_input = [NN_input, [x_traj(2:4,1:end-1);u'] ];
    NN_output = [NN_output, [x_traj(3,2:end);x_traj(4,2:end)] ];
end


function [H, H_real, RMSE] = calculate_H_RMS(iter_max,A_store,B_store,A_real_store,B_real_store)
% Calculate H:
H_real = B_real_store{1};
for iter = 2:iter_max
    H_real = [A_real_store{iter}*H_real, B_real_store{iter}];
end

H = B_store{1};
for iter = 2:iter_max
    H = [A_store{iter}*H, B_store{iter}];  
end

RMSE = sqrt(mean((H(:)-H_real(:)).^2));
end

function RMSE = calculate_H_RMS_sim(iter_max,A_store,B_store,A_real_store,B_real_store)
% Calculate H:
H_real = B_real_store{1};
for iter = 2:iter_max
    H_real = [A_real_store{iter}*H_real, B_real_store{iter}];
end

H = B_store{1};
for iter = 2:iter_max
    H = [A_store{iter}*H, B_store{iter}];  
end

RMSE = sqrt(mean((H(:)-H_real(:)).^2));
end








