%% first code for CSC in spatial domain
function [ d, z ]  = OnlineStochasticCSC(b, test_b, d_ind, kernel_size, lambda)

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
    outer_ite = 50; % num_image;
    max_it_z = 20;
    max_it_d = 10;
    
    batch_size = 1;    % change in batch mode
    total_time = 0;
    rho_d = 10;
    rho_z = 10;
    hit_rate = 0.1;    % probability for choosing one specific code 
    
    u_d     = zeros( k*k*num_kernel, 1 );
    x_hat_d = zeros( k*k*num_kernel, 1 );
    z       = zeros( n*n*num_kernel, batch_size );
    
%     ind = 1:1:n*n*num_kernel;
%     [ind1_all, ind2_all, v_all] = find( d_ind(:,ind) );

%     sequence = randperm(outer_ite*batch_size);
    sequence = 1:50;
    test_ite = 0;
    
    for ite = 1:outer_ite
        %% test set objective
%         if ite==1 || ite==outer_ite || mod(log2(ite),1) == 0
%             test_ite = test_ite + 1;
% %             tic
% %             D = sparse( ind1_all, ind2_all, d(v_all), n*n, n*n*num_kernel );
% %             if ite == 1
% %                 [z_test, x_hat_test, u_test, obj(test_ite), ~] = lasso(D, ...
% %                     reshape( test_b, [], size(test_b,3) ), []    , []        ,...
% %                     []    , 1, 10, 1.8, 50, 1, 1);
% %             else
% %                 [z_test, x_hat_test, u_test, obj(test_ite), ~] = lasso(D, ...
% %                     reshape( test_b, [], size(test_b,3) ), z_test, x_hat_test,...
% %                     u_test, 1, 10, 1.8, 50, 1, 1);
% %             end
% %             toc
% 
%             tic
%             if ite == 1
%                 [z_test, d_Z, Dz, obj(test_ite)] = lassoFFT(test_b, []      ,...
%                     [] , d, kernel_size, 1, 1, 50, 1e-3 );
%             else
%                 [z_test, d_Z, Dz, obj(test_ite)] = lassoFFT(test_b, z_test,...
%                     d_Z, d, kernel_size, 1, 1, 50, 1e-3 );
%             end
%             toc
%             fprintf( 'test objective: %10.4f; data fitting: %10.4f \n', ...
%                 obj(test_ite), obj(test_ite)- norm( 1*z_test(:),1) );
%         end
        
        %% compute codes
        t_train = tic;
        cur_image = sequence( mod( (ite-1)*batch_size+1, num_image )  : ... 
                              mod( ite*batch_size, num_image ) );
                      
        tt = tic;
        for j=1:numel(cur_image)
            
            ind = randperm( n*n*num_kernel, hit_rate*n*n*num_kernel );
            
            [ind1, ind2, v] = find( d_ind(:,ind) );
            
            D = sparse( ind1, ind2, d(v), n*n, numel(ind) );
            
            [z_tmp, ~, ~] = lasso(D, b(:,cur_image(j)), [], ...
                [], [], lambda, rho_z, 1.8, max_it_z, 2, 1);
            
            z(:,j) = zeros(n*n*num_kernel,1);
            z(ind,j) = z_tmp;
        end
        toc(tt)
        
        %% update dictionary
        tt = tic;
        Z = constructZ( z, n, batch_size, k, num_kernel );
        
        if (ite==1)
            B = Z'*reshape( b(:,cur_image), [],1 );
        else
            B = B_old * (ite-1) / ite + ...
                Z'*reshape( b(:,cur_image), [],1 ) / ite ;
        end
        
        if (ite==1) C = Z'*Z;
        else        C = C_old * (ite-1) / ite + Z'*Z / ite ;
        end
        
        d_old = d;
        [d, x_hat_d, u_d] = dicUpdate(C, B, d_old, x_hat_d, u_d, rho_d, 1.8, k*k, num_kernel, max_it_d);
        fprintf('%3d: dictionary updates: %10.4f\n', ite, norm( d-d_old ) );
        
        B_old = B;
        C_old = C;
        toc(tt)
        
        total_time = total_time + toc(t_train);
    end
 
    fprintf( 'total executioin time: %10.4f\n', total_time );
    
return;

function Z = constructZ( z, n, num_image, k, num_kernel )
    Z = spalloc( n*n*num_image, k*k*num_kernel, k*k*k*k*num_kernel*num_image );
    
    for i=1:num_image
        for j=1:num_kernel
            
            tmp_z = z( n*n*(j-1)+1:n*n*j , i );
            ind = find( abs(tmp_z)>0.01 );  %  ~= 0 );
            full_M = spalloc( n*n, k*k, k*k*length(ind) );
            
            for i1 = 1:length(ind)
                tmp = spdiags(ones(n,1)*tmp_z(ind(i1)), ...
                    floor((k+1)/2)-mod(ind(i1)-1,n)-1, n, k);
                M = kron( spdiags(ones(n,1), 0, k, k), tmp );
                
                ind1 = ceil(ind(i1)/n)-1-floor((k-1)/2);
                if ( ind1+k > n )
                    full_M( ind1*n+1:end, : ) = full_M( ind1*n+1:end, : ) + M( 1:(n-ind1)*n, : ); 
                elseif( ind1 < 0)
                    full_M( 1:(k+ind1)*n, : ) = full_M( 1:(k+ind1)*n, : ) + M(-ind1*n+1:end, : );
                else
                    full_M( ind1*n+1:(ind1+k)*n, : ) = full_M( ind1*n+1:(ind1+k)*n, : ) + M;
                end
            end
            
            Z( n*n*(i-1)+1 : n*n*i, k*k*(j-1)+1 : k*k*j ) = full_M;
        end
    end
return;