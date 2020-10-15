function [Phi,Psi_p,JPhi] = compute_Phi_and_JPhi(p,F,xi,dt)
Id = [];

for i = 1:length(xi)
    Id = [Id; xi(i)];
end

J = eye(length(xi));  % Identity 
Psi = Id;             % [xi1; xi2; xi3; xi4; xi5] 

Phi_syms = Psi;       % [xi1; xi2; xi3; xi4; xi5] 
JPhi_syms = J;        % Identity 

for k_deg = 1:p
    tic
    
    Psi = J*F;
    Psi = simplify(Psi); % Calculate Psi from eq(1)
    
      J = jacobian(Psi,xi); 
      J = simplify(J);   % Jacobian of Psi 
      
    % Taylor approxi. of the flow (discretization purpose) 
    Phi_syms = Phi_syms + (dt^k_deg/factorial(k_deg))*Psi;
    % Taylor approxi. of the jacobian of flow (linearization purpose)
    JPhi_syms = JPhi_syms + (dt^k_deg/factorial(k_deg))*J;
   
    toc
end
% Turn symbolic expressions into functions 
Phi = matlabFunction(simplify(Phi_syms),'Vars',[dt xi]);  % The flow 
Psi_p = matlabFunction(simplify(Psi),'Vars',xi);          % The last Psi 
JPhi = matlabFunction(simplify(JPhi_syms),'Vars',[dt xi]);% Jacobian of the flow

end







