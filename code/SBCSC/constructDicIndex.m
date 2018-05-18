function d_ind = constructDicIndex( kernel_size, col, row, d )
    
    k = kernel_size(1);
    num_kernel = kernel_size(3);
    m = col; 
    n = row;

    d_ind = spalloc( m*n, m*n*num_kernel, k*k*m*n*num_kernel );
    M = spalloc( m*n, m*n, k*k*m*n );
    
    if nargin == 3
        d = [1:prod(kernel_size)]';
    end
    
    for j=1:num_kernel
        tmp_d = d( (j-1)*k*k+1 : j*k*k )'; 
        for i=1:k           
            tmp_M = repmat( tmp_d( k*i:-1:(k*(i-1)+1) ), [m 1] );
            tmp_M = spdiags(tmp_M, -floor(k/2):floor(k/2), m, m);
            if( i==1 )
                M = kron( spdiags(ones(n,1), ...
                    ceil(k/2)-i, n, n), tmp_M);
            else
                M = M + kron( spdiags(ones(n,1), ...
                    ceil(k/2)-i, n, n), tmp_M);
            end
        end
        d_ind(:, (m*n*(j-1)+1):m*n*j) = M;
    end

end