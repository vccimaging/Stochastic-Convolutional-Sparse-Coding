function [z, x_hat, u, obj] = l2proj(A, b, d, x_hat_d, u_d, code, beta, rho, alpha,...
                                    k, num_kernel, MAX_ITER)

QUIET    = 1;
% MAX_ITER = 100;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;

x = zeros(n,1);
% x_hat = zeros(n,1);
% x = d;
x_hat = x_hat_d;
z = d;
u = u_d;

% cache the factorization
A_cal = @(p) A'*(A*p) + rho.*p;

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

t_start = tic;
for i = 1:MAX_ITER

    % x-update
     q = Atb + rho*(z - u);    % temporary value

    x = conj_grad(q, A_cal, x_hat, 100, 1e-5, 2);

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = proj_l2(x_hat + u, k, num_kernel);
    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(i)  = objective(A, b, beta, code, z);

    history.r_norm(i)  = norm(x - z, 'fro');
    history.s_norm(i)  = norm(-rho*(z - zold), 'fro');

    history.eps_pri(i) = sqrt(n)*ABSTOL + RELTOL*max(norm(x, 'fro'), norm(-z,'fro'));
    history.eps_dual(i)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u, 'fro');

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', i, ...
            history.r_norm(i), history.eps_pri(i), ...
            history.s_norm(i), history.eps_dual(i), history.objval(i));
    end

    if (history.r_norm(i) < history.eps_pri(i) && ...
       history.s_norm(i) < history.eps_dual(i))
         break;
    end

end

obj =  history.objval(i);

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, beta, code, z)
%     p = ( 1/2*norm(A*z - b,'fro').^2 + norm(lambda.*z(:),1) );
    p = ( 1/2*norm(A*z - b,'fro').^2  + norm(beta.*code(:),1) );
end

function z = proj_l2(x, k, num_kernel)
    z = zeros(size(x));
    for i=1:num_kernel
       z((i-1)*k+1:i*k) =  x((i-1)*k+1:i*k) ./ max( norm(x((i-1)*k+1:i*k)), 1);
    end
end