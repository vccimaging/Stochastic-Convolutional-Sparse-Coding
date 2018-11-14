function d = BCDproj( Z, b, d_old, k, num_kernel, MAX_ITER )
    
    d = d_old;
    r = b - Z*d_old;
    
    lip = zeros( num_kernel, 1);
    for i=1:num_kernel
        ind = (i-1)*k+1 : i*k;
        lip(i) =  norm(  full(Z(:,ind)'* Z(:,ind)) );
    end
    
    for i=1:MAX_ITER
       for j=1:num_kernel
           ind = (j-1)*k+1 : j*k;
           h = Z(:, ind )'*r / lip(j) ;           
           d_new = d(ind) + h;
           d_new = d_new ./ max( norm(d_new), 1 );          
           r = r + Z(:, ind)*( d(ind) - d_new );
           d(ind) = d_new;
       end
%        fprintf('%10.4f\n', norm(Z*d-b) );
    end
end