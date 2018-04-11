r%%%%%%%%%%%%%%%%%% 3X3 %%%%%%%%%%%%%%%%%%%% 
B = repmat((3:-1:1),[6 1])
A = full(spdiags(B, -1:1, 6,6))
M1 = kron( spdiags(ones(6,1), 1, 6, 6), A)

B = repmat((6:-1:4),[6 1])
A = full(spdiags(B, -1:1, 6,6))
M2 = kron( spdiags(ones(6,1), 0, 6, 6), A)

B = repmat((9:-1:7),[6 1])
A = full(spdiags(B, -1:1, 6,6))
M3 = kron( spdiags(ones(6,1), -1, 6, 6), A)

full(M1+M2+M3);

%%%%%%%%%%%%%%%%% nXn %%%%%%%%%%%%%%%%%%%%
k=5, n=10;
M = cell(k,1);
for i=1:k
    tmp_M = repmat( k*i:-1:(k*(i-1)+1), [n 1] );
    tmp_M = spdiags(tmp_M, -floor(k/2):floor(k/2), n, n);
    M{i} = kron( spdiags(ones(n,1), ceil(k/2)-i, n, n), tmp_M);
end

full_M = M{1};
for i=2:k
full_M = full_M + M{i};
end

%%%%%%%%%%%%%%% general generating D %%%%%%%%%%%%%%%%%%
z = reshape( PSF{1}(:,:,1), 1, [] );
k = length( PSF{1}(:,1,1) );
n = 10;

for i=1:k
    tmp_M = repmat( z( k*i:-1:(k*(i-1)+1) ), [n 1] );
    tmp_M = spdiags(tmp_M, -floor(k/2):floor(k/2), n, n);
    M{i} = kron( spdiags(ones(n,1), ceil(k/2)-i, n, n), tmp_M);
end
full_M = M{1};
for i=2:k
full_M = full_M + M{i};
end

%%%%%%%%%%%%%%% general generating Z %%%%%%%%%%%%%%%%%%
k = 11;
n = 100;

for i=1:n
    tmp_M = repmat( ones(k,1), [1 n] );
    tmp_M = spdiags(tmp_M, floor((k-1)/2)-n+1:floor((k-1)/2), n, k);
    M{i} = kron( spdiags(ones(n,1), floor((k-1)/2)-i+1, n, k), tmp_M );
end
full_M = M{1};
for i=2:n
full_M = full_M + M{i};
end

ind = 8;
tmp = spdiags(ones(n,1), floor((k+1)/2)-mod(ind-1,n)-1, n, k);
M = kron( spdiags(ones(n,1), floor((k-1)/2)-ceil(ind/n)+1, n, k), tmp );
full(M)
%%
ind = 35;
tmp = spdiags(ones(n,1), floor((k+1)/2)-mod(ind-1,n)-1, n, k);
M = kron( spdiags(ones(n,1), 0, k, k), tmp );

full_M = spalloc( n*n, k*k, 10 );

cur = ceil(ind/n)-1-floor((k-1)/2);
if ( cur+k > n )
    full_M( cur*n+1:end, : ) = M( 1:(n-cur)*n, : ); 
elseif( cur < 0)
    full_M( 1:(k+cur)*n, : ) = M(-cur*n+1:end, : );
else
    full_M( cur*n+1:(cur+k)*n, : ) = M;
end

full(full_M)
%%%%%%%%%%%%%%%%%%%%%% plot filters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
for i = 1:5
        D = constructD( d(:), n, k, num_kernel );
        z = lasso(D, b(:,1), 0.001, 1, 1.8);       
        Z = constructZ( z, n, 1, k, num_kernel );
        d = l2proj(Z, b(:,1), 0.1, 1, 1.8, k*k, num_kernel);
        
        dic = reshape( d, 11,11,100 );

        figure
        sub_dic = cell(size(dic,3),1 );
        for i=1:size(dic,3)
           sub_dic{i} = dic( :,:,i );
        end   
        for i=1:size(dic,3)
            subplot('Position',[(mod(i-1,10))/10 1-(ceil(i/10))/10 1/10-0.01 1/10-0.01]);
            imshow( sub_dic{i}, [], 'init', 'fit' );
        end
end


figure
imshow( reshape( b(:,1), 100, 100 ), [], 'init', 'fit' );
figure
imshow( reshape(D*z, 100, 100 ), [], 'init', 'fit' );

max_d = max(d(:)); min_d = min(d(:));
dic = reshape( d, 11,11,100 );
figure
sub_dic = cell(size(dic,3),1 );
for i=1:size(dic,3)
    sub_dic{i} = dic( :,:,i );
end
for i=1:size(dic,3)
    subplot('Position',[(mod(i-1,10))/10 1-(ceil(i/10))/10 1/10-0.01 1/10-0.01]);
    imshow( sub_dic{i}, [min_d max_d], 'init', 'fit' );
end

dic_z = reshape( z, 100, 100, 100);
Dz = zeros(100,100);
for i=1:100
   Dz = imfilter( dic_z(:,:,i), dic(:,:,i) ) + Dz;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%% verification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dz = zeros(100,100);
for i=1:100
Dz = conv2( dic_z(:,:,i), dic(:,:,i),'same' ) + Dz;
end
figure
imshow( Dz, [], 'init', 'fit' );
%%%%%%%%%%%%%%%%%%%%%%%%% felix code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = zeros(110,110);
for i=1:100
X = conv2( z(:,:,i,1), d(:,:,i),'same' ) + X;
end
figure
imshow( X, [], 'init', 'fit' );

figure; imshow( conv2( X, d(:,:,1),'same' ),[], 'init', 'fit' );

figure;
imshow( ifft2( fft2(X, 120,120) .* fft2( d(:,:,1), 120, 120 ) ) ,[], 'init', 'fit' );

figure; imshow( z( :,:,1,1 ),[], 'init', 'fit' );


%%%%%%%%%%%%%%%%%%%%%%% verify sparsity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_tmp = reshape( d, 11, 11, 100);
b_tmp = reshape( b(:,1), 100, 100);

corre = conv2( b_tmp, d_tmp(:,:,1),'same' );
figure; imshow( corre, [] );

array = nth_element( abs(corre(:)), 9000)';
pivot = array(9000);
corre( find(abs(corre)<pivot) ) = 0;

figure; imshow( corre, [] );


%%%%%%%%%%%%%%%%%%%%%% code histogram %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z_his = z(:,1);
figure; h = histogram( z_his( find(z_his~=0)) )


%%%%%%%%%%%%%%%%%%%%% psnr evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psnr( D*z(:,1), b(:,1))

%%%%%%%%%%%%%%%%%%%%% test objective %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x =  1:10;
% x1 = 
z1 = 1.0e+03 *[7.9687 5.6355 5.5438 5.5379    5.5134    5.5071    5.5117    5.4943    5.5012    5.5063];
% x2 = 
z2 =  1.0e+03 * [7.9056    5.6682    5.5850    5.4816    5.4595    5.4201    5.4124    5.4101    5.4020    5.3878];
% x3 = 
z3 = 1.0e+04 * [0.7776    1.1563    1.1727    1.0971    0.9316    0.6397    0.5874    0.5790    0.5572    0.5407];
figure
plot( 0:22:220-22, z1, '--o','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 0:80:720, z2, '--d','MarkerSize', 12, 'LineWidth',2);
hold on
plot( 0:75:675, z3, '--^','MarkerSize', 12, 'LineWidth',2 );
hold on
plot( 0:70:630, z4(1:10), '--*','MarkerSize', 12, 'LineWidth',2 );

set(gca,'fontsize',28)
set(gca,'xtick',[1 5 10]);
xlabel('Iteration', 'fontsize',28 );
set(gca, 'XLim', [1 10]);
set(gca, 'YLim', [5000 8000]);
set(gca,'ytick',[ 5000 6000 7000 8000]);
ylabel('Test objective', 'fontsize',28 );

legend('SOCSC', 'SBCSC', 'OCSC', 'FFCSC' );
set(gca, 'XScale', 'log');

% proposed 20 iterations inner:10  lambda:1.5 rho: 20
z1 =    1.0e+03 * [7.9421 5.6423 5.5613 5.5552 5.5291 5.5086 5.4949 5.4948 ...
    5.5011 5.4909 5.4930 5.4924 5.4634 5.4615 5.4596 5.4650 5.4613 5.4593 5.4545 5.4587];
% without active set 10 iterations:
1.0e+03 * [7.9056    5.6682    5.5850    5.4816    5.4595    5.4201    5.4124    5.4101    5.4020    5.3878];


% online randperm 0.1 220s
z1 = 1.0e+03 * [7.8721    5.7361    5.5679    5.5226    5.4816    5.4618    5.4288    5.4151    5.3981    5.3828];
% batch randperm 0.1 800s
z2 = 1.0e+03 * [7.8114    5.7424    5.5458    5.4376    5.3673    5.3325    5.3093    5.2924    5.2874    5.2772];
% Liu OCSC 750s
z3 = 1.0e+03 * [7.7460    6.2336    5.8762    5.7110    5.6589    5.5810    5.5595    5.5074    5.4848    5.4624];
% felix FFCSC 800s
z4 = 1.0e+03 * [7.8116    7.4212    8.8024    6.8605    7.8430    5.9489    5.9039    5.9213    5.6927    5.5132    5.4100    5.3384    5.2871    5.2563];
