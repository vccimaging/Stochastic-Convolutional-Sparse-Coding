%% online code for CSC in spatial domain
function [ d, z ]  = OnlineStochasticCSC(b, d_ind, z_ind, kernel_size, lambda)

    k = kernel_size(1);           
    num_kernel  = kernel_size(end);

    n = size(b,1);
    num_image = size(b,3);
    
    b = reshape(b, [], num_image);

    d = randn(k, k, num_kernel);
    % normalize the initialization. Not necessary but suggested.
    for i=1:num_kernel
       d(:,:,i) =  d(:,:,i) ./ max( norm( d(:,:,i), 'fro' ), 1);
    end
    d = d(:);
    
    %% set up parameters
    outer_ite = 10;  % num_image;
    max_it_z = 10;
    max_it_d = 10;
    
    batch_size = 1;
    rho_d = 50*lambda;
    rho_z = 10*lambda;
    hit_rate = 0.1;    % probability for choosing one specific code 
    gamma = 1;         % forget factor
    total_time = 0;
    
    % make sure outer_ite*batch_size not exceed the number of images
    sequence = randperm(outer_ite*batch_size);
    
    for ite = 1:outer_ite

        %% compute sparse codes
        t_train = tic;
        
        cur_image = sequence((ite-1)*batch_size+1:ite*batch_size);
        
        % randomly pick the sparse code for reconstruction
        ind = randperm( n*n*num_kernel, hit_rate*n*n*num_kernel );
        
        % construct (DM^T)
        [ind1, ind2, v] = find( d_ind(:,ind) );        
        D = sparse( ind1, ind2, d(v), n*n, numel(ind) );
        
        % if batch_size is large, use Cholesky decomposition
        % otherwise use conjugate gradient.
        % this is important for an efficient compute especially when
        % batch_size is large.
        z_tmp = lasso(D, b(:,cur_image), lambda, rho_z, 1.8, max_it_z, 2);
        
        % project the results onto its original domain
        z = zeros(n*n*num_kernel, batch_size);
        z(ind,:) = z_tmp;
        
        %% update dictionary       
        Z = constructZ( z, z_ind, n, batch_size, k, num_kernel );
           
        % compute surrogate matrices
        theta = (1-1/ite).^gamma;
        if (ite==1)
            B = Z'*reshape( b(:,cur_image), [],1 ); 
        else
            B = B_old * theta + (1-theta) * Z'*reshape( b(:,cur_image), [],1 ); 
        end
        
        if (ite==1) C = full(Z'*Z);
        else        C = C_old * theta + full(Z'*Z)*(1-theta) ;
        end
        
        d_old = d;
        d = dicUpdate(C, B, d_old, rho_d, 1.8, k*k, num_kernel, max_it_d);
        fprintf('ite: %3d - dictionary updates: %10.4f\n', ite, norm( d-d_old ) );
        
        B_old = B;
        C_old = C;
        
        total_time = total_time + toc(t_train);
    end
    
    save(sprintf('../filters/d_mini%2d_online%3d_P%1.1f.mat', ...
        batch_size, num_kernel, hit_rate ), 'd');
    fprintf( 'total executioin time: %10.4f\n', total_time );   
return;

function Z =constructZ( z, z_ind, n, num_image, k, num_kernel )
    ind1_all = zeros( k*k*length(find(z~=0)),1 );
    ind2_all = zeros( k*k*length(find(z~=0)),1 );
    v_all    = zeros( k*k*length(find(z~=0)),1 );
    cur = 1;
    for i=1:num_image
        for j=1:num_kernel
            tmp_z = z( n*n*(j-1)+1:n*n*j , i );
            ind = find( abs(tmp_z)>0.01 ); % or ~=0. either is ok
            for i1 = 1:length(ind)
                len = length( z_ind{ind(i1)}.ind1 );
                ind1_all( cur : cur+len-1 ) = z_ind{ind(i1)}.ind1 + n*n*(i-1);
                ind2_all( cur : cur+len-1 ) = z_ind{ind(i1)}.ind2 + k*k*(j-1);
                v_all(    cur : cur+len-1 ) = z( ind(i1) + n*n*(j-1), i );
                cur = cur + len;
            end
        end
    end
    Z = sparse( ind1_all(1:cur-1), ind2_all(1:cur-1), v_all(1:cur-1),...
        n*n*num_image, k*k*num_kernel );
return;