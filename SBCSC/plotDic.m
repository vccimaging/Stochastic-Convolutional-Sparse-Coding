%% online code for CSC in spatial domain
function plotDic(d, s, max_d, min_d)

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
    
end