function jac = NN_jacob(net,xu_nom, x1_step1,y1_step1)

xoffset = x1_step1.xoffset;
xmin = x1_step1.ymin;
xgain = x1_step1.gain;
wi = net.IW{1,1};  % Input weight matrices
% wo = net.LW{2,1};  % Layer weight matrices
bi = net.b{1,1};   % Input Bias vectors
% bo = net.b{length(net.LW)-1,1};   % Output Bias vectors
ygain = y1_step1.gain;

% computation of critical values
z{1} = wi*(xgain.*(xu_nom-xoffset)+xmin)+bi;
zPr{1} = 1-tansig(z{1}).^2;
for i = 2:length(net.LW)-1
    z{i} = net.LW{i,i-1}*tansig(z{i-1})+net.b{i,1};
    zPr{i} = 1-tansig(z{i}).^2;
end

% computation of Jacobian based on chain role
mul_comp = net.LW{2,1}*diag(zPr{1});
for i = 2:length(net.LW)-1
    mul_comp = net.LW{i+1,i}*diag(zPr{i})*mul_comp;
end


jac = diag(1./ygain)*mul_comp*wi*diag(xgain);


end