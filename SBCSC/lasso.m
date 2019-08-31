function z = lasso(A, b, lambda, rho, alpha, MAX_ITER, option)

[p, n] = size(A);
[~, m] = size(b);

% save a matrix-vector multiply
Atb = A'*b;

x = zeros(n,m);
z = zeros(n,m);
x_hat = zeros(n,m);
u = zeros(n,m);

% cache the factorization
if option == 1 || isempty( option )
    [L U] = factor(A, rho);
elseif option == 2
    A_cal = @(p) A'*(A*p) + rho.*p;
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

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);
    % u-update
    u = u + (x_hat - z);
    
end

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
