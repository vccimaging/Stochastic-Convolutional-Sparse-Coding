function [ z, d_Z, Dz, obj ] = lassoFFT(b, z, d_Z, d, kernel_size, lambda_residual, ...
    lambda_prior, max_it_z, tol )
       
    %Timing
    tstart = tic;

    d = reshape( d, kernel_size );
    psf_s = kernel_size(1);
    k = kernel_size(3);
    n = size(b,3);
                
    %PSF estimation
    psf_radius = floor( psf_s/2 );
    size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, n];
    size_z = [size_x(1), size_x(2), k, n];
    
    objective = @(z, dh) objectiveFunction( z, dh, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x );
    
    %Prox for masked data
    [M, Mtb] = precompute_MProx(b, psf_radius); %M is MtM
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); 
    
    %Prox for sparsity
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
       
    %% Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
%     gamma_heuristic = 60 * lambda_prior * 1/max(b(:));
%     gammas_Z = [gamma_heuristic / 1, gamma_heuristic]; %[gamma_heuristic / 500, gamma_heuristic];
    gammas_Z = [1, 20];

    d = padarray( d, [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');
    d = circshift(d, -[psf_radius, psf_radius, 0] );
    d_hat = fft2(d);
    
    %% Initialize variables for Z
    varsize_Z = {size_x, size_z};
    xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    %Initial iterates
    if isempty(z)
        z = zeros(size_z);
        z_hat = zeros(size_z);
    else
        z_hat = reshape(fft2(reshape(z, size_z(1), size_z(2), [])), size_z);
    end
    if isempty(d_Z)
        d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    end
    if isempty(tol)
        tol = 1e-3;
    end
    
        %% compute sparsity term
        
        %Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);
        dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] ); %Same for all images
              
        for i_z = 1:max_it_z

            %Compute v_i = H_i * z
            v_Z{1} = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* z_hat, 3), size_x) ));
            v_Z{2} = z;

            %Compute proximal updates
            u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
            u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );

            for c = 1:2
                %Update running errors
                d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});

                %Compute new xi and transform to fft
                xi_Z{c} = u_Z{c} + d_Z{c};
                xi_Z_hat{c} = reshape( fft2( reshape( xi_Z{c}, size_x(1), size_x(2), [] ) ), size(xi_Z{c}) );
            end

            %Solve convolutional inverse
            % z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            zold = z;
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z);
            z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2),[]) )), size_z );
      
            z_diff = z - zold;
            if norm(z_diff(:),2)/ norm(z(:),2) < tol
                break;
            end
        end
  
        zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
        Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* zhat, 3), size_x) ));
        Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);
               
        obj = objective(z, d_hat);
        fprintf('iteration: %3d --> Obj %3.3g \n', i_z, obj );
        toc(tstart);
        
%         z = z(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:,:);    
return;

function [M, Mtb] = precompute_MProx(b, psf_radius)
    
    M = padarray(ones(size(b)), [psf_radius, psf_radius, 0], 0, 'both');
    Mtb = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');
    
return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z )


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    
    %Rho
    rho = gammas(2)/gammas(1);
    
    %Compute b
    b = dhatT .* permute( repmat( reshape(xi_hat{1}, sy*sx, 1, n), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(xi_hat{2}, sy*sx, k, n), [2,1,3] );
    
    %Invert
    scInverse = repmat( ones([1,sx*sy]) ./ ( rho * ones([1,sx*sy]) + dhatTdhat.' ), [k,1,n] );
    x = 1/rho *b - 1/rho * scInverse .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    
    %Final transpose gives z_hat
    z_hat = reshape(permute(x, [2,1,3]), size_z);

return;

function f_val = objectiveFunction( z, d_hat, b, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

    %Dataterm and regularizer
    zhat = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,n]) .* zhat, 3), size_x) ));
    
    f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;
