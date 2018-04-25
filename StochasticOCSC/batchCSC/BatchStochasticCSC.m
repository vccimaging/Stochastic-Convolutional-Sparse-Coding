%% first code for CSC in spatial domain
function [ d, z, obj ]  = BatchStochasticCSC(b, test_b, d_ind, z_ind, kernel_size, lambda)

    k = kernel_size(1);           
    num_kernel  = kernel_size(end);

    n = size(b,1);
    num_image = size(b,3);
    
    b = reshape(b, [], num_image);

    d = randn(k, k, num_kernel);
    for i=1:num_kernel
       d(:,:,i) =  d(:,:,i) ./ max( norm( d(:,:,i), 'fro' ), 1);
    end
    back_d = d(:);
    d = back_d;
    
    %% parameters
    outer_ite = 10;
    max_it_z = 20;
    max_it_d = 10;
    
    batch_size = 10;
    total_time = 0;
    rho_d = 10;
    rho_z = 1;
    hit_rate = 0.1;    % probability for choosing one specific code 
    
    u_d     = zeros( k*k*num_kernel, 1 );
    x_hat_d = zeros( k*k*num_kernel, 1 );
    z       = zeros( n*n*num_kernel, batch_size );
    
%     ind = 1:1:n*n*num_kernel;
%     [ind1_all, ind2_all, v_all] = find( d_ind(:,ind) );

%     test_ite = 0;
    cur_image = 1:num_image; 
    for ite = 1:outer_ite
        %% test set objective
%          if ite==1 || ite==outer_ite || mod(log2(ite),1) == 0
%           test_ite = test_ite + 1;
%           [z_test, d_Z, Dz, obj(test_ite)] = lassoFFT(test_b, []      ,...
%                   [] , d, kernel_size, 1, 1, 50, 1e-3 );
%            fprintf( 'test objective: %10.4f; non-zero elements: %6d, %6d \n', ...
%              obj(test_ite), size( find( abs(z_test(:))>0.01),1 ),size( find( abs(z_test(:))>0.1),1 ) );
%            if ite>30
%               save(sprintf('filter/d%4d_mini%2d_online%3d_forget%1.1f_P%1.1f_lambda%1.1f_rho%2.1f_ite%2d_rho_%2.1f_%2d.mat',...
%                   num_image, batch_size, num_kernel, gamma, hit_rate, lambda, ...
%                   rho_z, max_it_z, rho_d, ite ), 'd'); 
%            end
%          end
        %% compute codes
        t_train = tic;
                      
        tt = tic;         
            ind = randperm( n*n*num_kernel, hit_rate*n*n*num_kernel );           
            [ind1, ind2, v] = find( d_ind(:,ind) );           
            D = sparse( ind1, ind2, d(v), n*n, numel(ind) );
%             D = sparse( ind1_all, ind2_all, d(v_all), n*n, numel(ind) );
            [z_tmp, ~, ~] = lasso(D, b(:,cur_image), [], ...
                [], [], lambda, rho_z, 1.8, max_it_z, 1, 1);

            z = zeros(n*n*num_kernel, batch_size);
            z(ind,:) = z_tmp;
%             z = z_tmp;
        toc(tt)
        
        %% update dictionary
        tt = tic;        
        Z = constructZ( z, z_ind, n, batch_size, k, num_kernel );
                 
        d_old = d;
        [d, x_hat_d, u_d, obj(ite)] = l2proj(Z, b(:), d, x_hat_d, u_d, z, lambda, ...
            rho_d, 1.8, k*k, num_kernel, max_it_d);
        fprintf('%3d: dictionary updates: %10.4f\n', ite, norm( d-d_old ) );
        toc(tt)
                total_time = total_time + toc(t_train);
    end
 
    fprintf( 'total executioin time: %10.4f\n', total_time );
    save(sprintf('filter/none/d_batch%3d_fruit%2d_P%1.2f_lambda%1.1f_rho%2d_ite%2d_rho_%2d.mat',...
        num_kernel, num_image,  hit_rate, lambda, ...
        rho_z, max_it_z, rho_d ), 'd');
return;

function Z =constructZ( z, z_ind, n, num_image, k, num_kernel )
    ind1_all = zeros( k*k*length(find(z~=0)),1 );
    ind2_all = zeros( k*k*length(find(z~=0)),1 );
    v_all    = zeros( k*k*length(find(z~=0)),1 );
    cur = 1;
    for i=1:num_image
        for j=1:num_kernel
            tmp_z = z( n*n*(j-1)+1:n*n*j , i );
            ind = find( abs(tmp_z)>0.01 );
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