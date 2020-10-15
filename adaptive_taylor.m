function [t_store, x_trajJ] = adaptive_taylor(p,Phi,Psi_p,interval,x0)

xJ = x0;
x_trajJ = [xJ];

err_tol = 10e-9;

t = interval(1);
t_store = [t];
dt_store = [];

while t < interval(2)
    T = num2cell(xJ);
    dt = min((err_tol*factorial(p)/norm(Psi_p(T{:}),inf))^(1/p),interval(2)-t);
    t = t+dt;
    t_store = [t_store t];
    dt_store = [dt_store, dt];
    
    xJ = Phi(dt,T{:});
    x_trajJ = [x_trajJ xJ];
    
end

t_store = transpose(t_store);
x_trajJ = transpose(x_trajJ);

end
