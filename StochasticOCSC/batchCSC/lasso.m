function [z, x_hat, u, obj, k] = lasso(A, b, z, x_hat_z, u_z, ...
    lambda, rho, alpha, MAX_ITER, option, QUIET)
t_start = tic;

if nargin < 11
    QUIET    = 0;
end
% MAX_ITER = 10;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[p, n] = size(A);
[~, m] = size(b);

% save a matrix-vector multiply
Atb = A'*b;

x = zeros(n,m);
if isempty(z)
    z = zeros(n,m);
end

if isempty(x_hat_z)
    x_hat_z = zeros(n,m);
end
x_hat = x_hat_z;

if isempty(u_z)
    u_z = zeros(n,m);
end
u = u_z;

% cache the factorization
if option == 1 || isempty( option )
    [L U] = factor(A, rho);
elseif option == 2
    A_cal = @(p) A'*(A*p) + rho.*p;
end

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update
    q = Atb + rho*(z - u);    % temporary value

    for i=1:m
        if option == 1 || isempty( option )  
            if( p >= n )    % if skinny
               x(:,i) = U \ (L \ q(:,i));
            else            % if fat
               x(:,i) = q(:,i)/rho - (A'*(U \ ( L \ (A*q(:,i)) )))/rho^2;
            end
        elseif option == 2
            x(:,i) = conj_grad(q(:,i), A_cal, x_hat(:,i), 100, 1e-5, 2);
        end
    end

    % faster than iteration
%      x = conj_grad(q, A_cal, x_hat, 100, 1e-4, 3);
   
    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);
    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z, 'fro');
    history.s_norm(k)  = norm(-rho*(z - zold), 'fro');

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x, 'fro'), norm(-z,'fro'));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u, 'fro');

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end

obj =  history.objval(k);

if ~QUIET
    toc(t_start);
end

end

function p = objective(A, b, lambda, x, z)
%     p = ( 1/2*sum((A*z - b).^2) + norm(lambda.*z(:),1) );
    p = ( 1/2*norm(A*z - b,'fro').^2 + norm(lambda.*z(:),1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end
