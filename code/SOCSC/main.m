addpath('../image_helpers');
CONTRAST_NORMALIZE = 'local_cn'; 
ZERO_MEAN = 1;   
COLOR_IMAGES = 'gray'; 
% read images
% we suggest to work on 100*100 (square) image patches (avoid working on large images and
% large number of filters at the same time)
[b] = CreateImages('../datasets/city_100_100',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); % Replace directory with large image data directory.
b = reshape(b, size(b,1), size(b,2), [] );

%% Define the parameters
kernel_size = [11, 11, 100];
lambda = 1;

%% cache the vector to matrix info for efficient indexing
% It may take some time (roughly 40s), while you only need to compute it once and store it
d_ind = constructDicIndex(  kernel_size, size(b,1), size(b,2) );
z_ind = constructCodeIndex( kernel_size, size(b,1) );

%% online stochastic CSC
[ d, z ]  = OnlineStochasticCSC(b, d_ind, z_ind, kernel_size, lambda);

%% visualize dictionary
plotDic( d, 10, max(d(:)), min(d(:)) );
