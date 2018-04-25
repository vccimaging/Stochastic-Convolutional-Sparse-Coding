%% test
% [b] = CreateImages('../../proposed_CSC/imageNet/testing',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); % Replace directory with large image data directory.
CONTRAST_NORMALIZE = 'local_cn'; 
% CONTRAST_NORMALIZE = 'none'; 
ZERO_MEAN = 1;   
COLOR_IMAGES = 'gray'; 
% [b] = CreateImagesJinhui('../../proposed_CSC/imageNet/testing',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); % Replace directory with large image data directory.
[b] = CreateImages('../../CCSC_code_ICCV2017-master/CCSC_code_ICCV2017-master/2D/Poisson_deconv/dataset_norm/1',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
test_b = reshape(b, size(b,1), size(b,2), [] );
signal = test_b(:,:,1);

D = constructDicIndex( kernel_size, size(test_b,1),size(test_b,2), d(:) );


%% Reconstruction from sparse data

%Sampling matrix
MtM = zeros(size(signal));
%MtM(1:2:end, 1:2:end) = 1;
MtM(rand(size(MtM)) < 0.5 ) = 1;

signal_sparse = signal;
signal_sparse( ~MtM ) = NaN;
% signal_fill = fillmiss(signal_sparse);
% offset = signal_fill - localContrast( signal_fill, ZERO_MEAN );

signal_sparse( ~MtM ) = min(signal_sparse(:));

ind = find( MtM(:) );
M = sparse( 1:numel(ind), ind, 1,  numel(ind),numel(MtM));

% M = spdiags(MtM(:), 0, numel(MtM),numel(MtM));

% [z, ~] = lasso( M*D, M*(signal_sparse(:)-offset(:)), [], [], [], 0.1, 3, 1.8, 50, 1, 0); 
% sig_rec = reshape( D*z, size(signal) ) + offset;
% MD = M*D;
% [z, ~] = lasso( MD, M*signal_sparse(:), [], [], [], 0.1, 3, 1.8, 50, 1, 0); 
% sig_rec = reshape( D*z, size(signal) );

tic();
[z, sig_rec] = sparseReconstruction(signal_sparse,...
    d, MtM, 5, 2, 50, 1e-3, signal, 'brief'); 
toc;

%Show result
figure(); 
subplot('Position',[0 0 1/3-0.01 1]),
imshow( signal, [], 'init', 'fit' ); title('Orig');
subplot('Position',[1/3 0 1/3-0.01 1]),
imshow( signal_sparse, [], 'init', 'fit' ); title('subsample');
subplot('Position',[2/3 0 1/3-0.01 1]),
imshow( sig_rec, [], 'init', 'fit' ); title('Reconstruction');
% subplot(1,3,1), imshow( signal, [], 'init', 'fit' ); axis image, colormap gray; title('Orig');
% subplot(1,3,2), imshow( signal_sparse, [], 'init', 'fit' ); axis image, colormap gray; title('subsample');
% % subplot(1,3,2), imshow( signal_sparse-offset, [], 'init', 'fit' ); axis image, colormap gray; title('subsample');
% subplot(1,3,3), imshow( sig_rec, [], 'init', 'fit' ); axis image, colormap gray; title('Reconstruction');

psnr( mat2gray(sig_rec), mat2gray(signal) )
ssim( mat2gray(sig_rec), mat2gray(signal) )

PSNR( signal, sig_rec )