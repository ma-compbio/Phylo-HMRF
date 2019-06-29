%%
function [state_mtx_vec,len_vec1] = read_state_test(state_vec, chrom_id, ...
    len_vec, color_vec1, n_iter, n_component, filename_idx, iter_id, output_path)

chrom = chrom_id;
b = find(len_vec(:,end)==chrom);
if isempty(b)
    fprintf('chromosome not found! %d\r\n',chrom);
    return;
end

len_vec1 = len_vec(b,:);
region_num = length(b);
state_mtx_vec = cell(region_num,1);

if ~exist(output_path, 'dir')
    mkdir(output_path);
end

for k1 = 1:region_num
    region_id = b(k1);
    
    window_size_1 = len_vec(region_id,4);
    window_size_2 = len_vec(region_id,5);
    
    output_filename1 = sprintf('%s/%d_chr%d_region%d_%d_%d.jpg',...
        output_path, filename_idx,chrom,region_id,window_size_1,iter_id);
    [~, color2] = color_map_sub(state_vec,color_vec1,...
        len_vec,iter_id,chrom,region_id,n_component,output_filename1);
    
    sel_id = 0;
    id1 = len_vec(region_id,2)+1;
    id2 = len_vec(region_id,3);
    
    window_size1 = 5;
    threshold = 80;
    if window_size_1<100
        threshold = 25;
    end
    % n_iter = 3;
    output_filename2 = sprintf('%s/%d_chr%d_region%d_%d_%d.jpg',...
        output_path, filename_idx,chrom,region_id,window_size_1,iter_id);
    mtx_1 = small_region_test(chrom,region_id,iter_id,color2,color_vec1,n_component,...
        window_size1,threshold,n_iter,sel_id,output_filename2);
    state_mtx = reshape(mtx_1,[window_size_1,window_size_2]);
    
    fprintf('%d %d %d %d %d\r\n',iter_id,chrom_id,region_id,id1,id2);
    
    state_mtx_vec{k1} = state_mtx;
end

end

