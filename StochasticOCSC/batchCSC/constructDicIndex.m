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


% function dic_index = constructDicIndex( kernel_size, image_size )
% 
%     k = kernel_size(1);
%     num_kernel = kernel_size(3);
%     n = image_size;
% 
%     D = spalloc( n*n, n*n*num_kernel, k*k*n*n*num_kernel );
%     M = spalloc( n*n, n*n, k*k*n*n );
%     
%     d = [1:prod(kernel_size)]';
%     
%     tic;
%     for j=1:num_kernel
%         for i=1:k
% %              tmp_d = reshape( d(:,:,j), 1, [] );
%             tmp_d = d( (j-1)*k*k+1 : j*k*k )'; 
%             tmp_M = repmat( tmp_d( k*i:-1:(k*(i-1)+1) ), [n 1] );
%             tmp_M = spdiags(tmp_M, -floor(k/2):floor(k/2), n, n);
%             if( i==1 )
%                 M = kron( spdiags(ones(n,1), ...
%                     ceil(k/2)-i, n, n), tmp_M);
%             else
%                 M = M + kron( spdiags(ones(n,1), ...
%                     ceil(k/2)-i, n, n), tmp_M);
%             end
%         end
%         D(:, (n*n*(j-1)+1):n*n*j) = M;
%     end
%     toc
%     
%     d_cell = num2cell(d);
%     
%     t_start = tic;
%     index_cell = cell( size(d,1) ,1);
%     for i=1:size(d,1)
%         index_cell{i} = find(D == i);
%     end
%     toc(t_start);
%     
%     t_start = tic;
%     index = cellfun(@(pos) find(D==pos), d_cell, 'UniformOutput', false);
%     toc(t_start);
%     
%     tic;
%     for i=1:18
%         [idx1,idx2] = ind2sub( size(D), index{i});
%         if( i==1 )
%             D_idx = sparse( idx1, idx2, 1, size(D,1), size(D,2));
%         else
%             D_idx = D_idx + sparse( idx1, idx2, i, size(D,1), size(D,2));
%         end
%     end
%     toc
%     
%     %%%%%%%%%%%%%%%%%%%%%%% without consider boundary %%%%%%%%%%%%%%%%%%%%%
%     
%     tic;
%     D_al = spalloc( n*n, n*n*num_kernel, k*k*n*n*num_kernel );
%     M = spalloc( n*n, n*n, k*k*n*n );
%     
%     for j=1:num_kernel
%         for i=1:k
% %              tmp_d = reshape( d(:,:,j), 1, [] );
%             tmp_d = d( (j-1)*k*k+1 : j*k*k )'; 
%             tmp_M = repmat( tmp_d( k*i:-1:(k*(i-1)+1) ), [n*n 1] );
%                     
%             if( i==1 )
%                 M = spdiags(tmp_M, n*(k-ceil(k/2)-i+1)-floor(k/2):...
%                             n*(k-ceil(k/2)-i+1)+floor(k/2), n*n, n*n);
%             else
%                 M = M + spdiags(tmp_M, n*(k-ceil(k/2)-i+1)-floor(k/2):...
%                             n*(k-ceil(k/2)-i+1)+floor(k/2), n*n, n*n);
%             end
%         end
%         D_al(:, (n*n*(j-1)+1):n*n*j) = M;
%     end
%     toc;
%     
% %     D_ind = spalloc( n*n, n*n*num_kernel, k*k*n*n*num_kernel );
%     
%     ind = 1:10:n*n*num_kernel;
%     [ind1, ind2, v] = find( D(:,ind) );
%     
%     tic;  
% %     dic = cell(n*n*num_kernel,1);
%     dic.ind1 = cell(n*n*num_kernel,1);
%     dic.ind2 = cell(n*n*num_kernel,1);
%     dic.v = cell(n*n*num_kernel,1);
%     for i=1:n*n*num_kernel
%         [dic.ind1{i}, dic.ind2{i}, dic.v{i}] = find( D(:,i) );
%     end
%     toc;
%     
%     tic;     %%% 1.77s
%     ind = (1:1:n*n*num_kernel)'; 
%     D_ind = sparse( cat(1,dic.ind1{ind}), cat(1,dic.ind2{ind}), ...
%         d( cat(1,dic.v{ind}) ), n*n, n*n*num_kernel );
%     toc;
%     
%     tic;    %%% 1.87s
%         ind = 1:10:n*n*num_kernel;
%         [ind1, ind2, v] = find( D(:,ind) );
%         D_ind = sparse( ind1, ind(ind2)', d(v), n*n, n*n*num_kernel );
%     toc
%     
% end