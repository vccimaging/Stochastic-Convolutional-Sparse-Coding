function z_ind = constructCodeIndex( kernel_size, n )
    
    k = kernel_size(1);
    z_ind = cell(n*n, 1);
    
    z = (1 : n*n);   
    M=0;
    for i=1:n
        tmp_M = repmat( z( (n*(i-1)+1):n*i ), [k 1] );
        tmp_M = spdiags(tmp_M, -floor(k/2):n-floor(k/2)-1, k, n);
        tmp_M = tmp_M';
        if( i==1 )
            M = sparse( kron( spdiags(ones(k,1), ...
                ceil(k/2)-i, n, k), tmp_M) );
        else
            M = M + sparse( kron( spdiags(ones(k,1), ...
                ceil(k/2)-i, n, k), tmp_M) );
        end
    end
    Z_ind = M;
    
    for i1 = 1:length(z)
        [ind1, ind2] = find( Z_ind==z(i1) );
        z_ind{i1}.ind1 = ind1;
        z_ind{i1}.ind2 = ind2;
    end

end