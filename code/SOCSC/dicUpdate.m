function z = dicUpdate(A, Atb, d, rho, alpha, k, num_kernel, MAX_ITER)

ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[~, n] = size(A);

if isempty(d)
    z = zeros(n,1);
else
    z = d;
end

x_hat = zeros(n,1);
u = zeros(n,1);

% [L U] = factor(A, rho);
A_cal = @(p) A*p + rho.*p;

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

    history.r_norm(i)  = norm(x - z, 'fro');
    history.s_norm(i)  = norm(-rho*(z - zold), 'fro');

    history.eps_pri(i) = sqrt(n)*ABSTOL + RELTOL*max(norm(x, 'fro'), norm(-z,'fro'));
    history.eps_dual(i)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u, 'fro');

    if (history.r_norm(i) < history.eps_pri(i) && ...
       history.s_norm(i) < history.eps_dual(i))
         break;
    end

end

end

function z = proj_l2(x, k, num_kernel)
    z = zeros(size(x));
    for i=1:num_kernel
       z((i-1)*k+1:i*k) =  x((i-1)*k+1:i*k) ./ max( norm(x((i-1)*k+1:i*k)), 1);
    end
end