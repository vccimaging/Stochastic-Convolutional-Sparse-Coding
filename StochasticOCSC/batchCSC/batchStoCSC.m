addpath('../image_helpers');
CONTRAST_NORMALIZE = 'none'; % 'local_cn'; 
ZERO_MEAN = 1;   
COLOR_IMAGES = 'gray'; 
% read training and testing images
% [b] = CreateImagesJinhui('../../proposed_CSC/imageNet/training',...
% CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); 
[b] = CreateImages('./dataset/fruit_100_100',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
 train_b = reshape(b, size(b,1), size(b,2), [] );

[b] = CreateImagesJinhui('../../proposed_CSC/imageNet/testing',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES); % Replace directory with large image data directory.
test_b = reshape(b, size(b,1), size(b,2), [] );

%% Define the parameters
kernel_size = [11, 11, 225];
lambda = 0.1;

d_ind = constructDicIndex(  kernel_size, size(train_b,1), size(train_b,2) );
z_ind = constructCodeIndex( kernel_size, size(train_b,1) );
%% online stochastic CSC
t_start = tic;
[ d, z, obj ]  = BatchStochasticCSC(train_b, [], d_ind, z_ind, kernel_size, lambda);
toc(t_start);

%% visualize dictionary
s = 15;
max_d = max(d(:)); min_d = min(d(:));
dic = reshape( d, 11,11,s*s );
figure
sub_dic = cell(size(dic,3),1 );
for i=1:size(dic,3)
    sub_dic{i} = dic( :,:,i );
end
for i=1:size(dic,3)
    subplot('Position',[(mod(i-1,s))/s 1-(ceil(i/s))/s 1/s-0.001 1/s-0.001]);
    imshow( sub_dic{i}, [min_d max_d], 'init', 'fit' );
end

%% Save
save(sprintf('d_online10_100_01.mat'), 'd_online10');
save(sprintf('d_online10_225_01.mat'), 'd_online10_225');
save(sprintf('d_online50_100_01.mat'), 'd_online50_100');
save(sprintf('d_online50_225_01.mat'), 'd_online50_225');