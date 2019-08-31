addpath('../image_helpers');

% read images
% we suggest to work on square image patches (avoid working on large images and
% large number of filters at the same time)
% CONTRAST_NORMALIZE = 'local_cn'; 
% ZERO_MEAN = 1;   
% COLOR_IMAGES = 'gray'; 
% [b] = CreateImages('../datasets/fruit_100_100',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
% b = reshape(b, size(b,1), size(b,2), [] );

load city;
% load fruit;

%% Define the parameters
kernel_size = [11, 11, 100];
lambda = 1;
hit_rate = 1;    % probability for choosing one specific code

%% cache the vector to matrix info for efficient indexing
% It may take some time (roughly 30s), while you only need to compute it once and store it
% d_ind = constructDicIndex(  kernel_size, size(b,1), size(b,2) );
% z_ind = constructCodeIndex( kernel_size, size(b,1) );
load d_ind; load z_ind;

%% online stochastic CSC
[ d, z ]  = OnlineStochasticCSC(b, d_ind, z_ind, kernel_size, hit_rate, lambda);

%% visualize dictionary
plotDic( d, 10, max(d(:)), min(d(:)) );
