function [z, x_hat, u, history] = dicUpdate(A, Atb, d, x_hat_d, u_d, rho, alpha,...
                                    k, num_kernel, MAX_ITER)
t_start = tic;

QUIET    = 0;
% MAX_ITER = 100;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[~, n] = size(A);

x = zeros(n,1);

if isempty(d)
    z = zeros(n,1);
else
    z = d;
end

if isempty(x_hat_d)
    x_hat_d = zeros(n,1);
end
x_hat = x_hat_d;

if isempty(u_d)
    u_d = zeros(n,1);
end
u = u_d;

% [L U] = factor(A, rho);
A_cal = @(p) A*p + rho.*p;

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for i = 1:MAX_ITER

    % x-update
     q = Atb + rho*(z - u);    % temporary value

%     x = U \ (L \ q);
    x = conj_grad(q, A_cal, x_hat, 100, 1e-5, 2);

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = proj_l2(x_hat + u, k, num_kernel);
    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
%     history.objval(i)  = objective(A, b, beta, x, z);

    history.r_norm(i)  = norm(x - z, 'fro');
    history.s_norm(i)  = norm(-rho*(z - zold), 'fro');

    history.eps_pri(i) = sqrt(n)*ABSTOL + RELTOL*max(norm(x, 'fro'), norm(-z,'fro'));
    history.eps_dual(i)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u, 'fro');

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', i, ...
            history.r_norm(i), history.eps_pri(i), ...
            history.s_norm(i), history.eps_dual(i));
    end

    if (history.r_norm(i) < history.eps_pri(i) && ...
       history.s_norm(i) < history.eps_dual(i))
         break;
    end

end

if ~QUIET
    toc(t_start);
end

end

% function p = objective(A, b, beta, x, z)
% %     p = ( 1/2*norm(A*z - b,'fro').^2 + norm(lambda.*z(:),1) );
%     p = ( 1/2*norm(A*z - b,'fro').^2 ) ;
% end

function z = proj_l2(x, k, num_kernel)
    z = zeros(size(x));
    for i=1:num_kernel
       z((i-1)*k+1:i*k) =  x((i-1)*k+1:i*k) ./ max( norm(x((i-1)*k+1:i*k)), 1);
    end
end

function [L U] = factor(A, rho)
    [~, n] = size(A);
    L = chol( A + rho*speye(n), 'lower' );
    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end