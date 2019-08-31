function d = BCDdicUpdate(C, B, d_old, k, num_kernel, MAX_ITER )
       
    d = d_old;

    lip = zeros( num_kernel, 1);
    for i=1:num_kernel
        ind = (i-1)*k+1 : i*k;
        lip(i) =  norm( C(ind, ind) );
    end

    for i=1:MAX_ITER
       for j=1:num_kernel
           ind = (j-1)*k+1 : j*k;
           h = ( B(ind) - C(ind,:)*d ) ./ lip(j) ;           
           d_new = d(ind) + h;
           d_new = d_new ./ max( norm(d_new), 1 );          
           d(ind) = d_new;
       end
    end

end