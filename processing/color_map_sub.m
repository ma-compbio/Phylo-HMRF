%% color_map_sub: generate color map based on estimated states
function [color1, color2] = color_map_sub(state_vec,color_vec1,len_vec,...
                           iter_id,chrom,region_id,n_component,output_filename1)
sel_id = 0;
smooth_id = 0;
for i = region_id:region_id
    id1 = len_vec(i,2)+1;
    id2 = len_vec(i,3);
    window_size = len_vec(i,4);
    window_size1 = len_vec(i,5);
    region_typeId = len_vec(i,end-1);
    % start_region = len_vec(i,end);
    start_region = len_vec(i,6);
    state_1 = state_vec(id1:id2); % read state of this region
    if smooth_id==1
        % state1 = state_smooth(state_1);
        state1 = state_smooth_unsym(state_1,window_size);
    else
        % state1 = state_1;
        if region_typeId==1
            t_state = zeros(window_size,window_size);
            sym_Idx = index_sym1(window_size,window_size);
            t_state(sym_Idx) = state_1;
            t_state = t_state';
            t_state(sym_Idx) = state_1;
            state1 = t_state(:);
        else
            state1 = reshape(state_1,window_size1,window_size); % reading by column
            state1 = state1';
            state1 = state1(:); % convert to reading by row
        end
    end
    window_size_1 = [window_size,window_size1];
    [color1,color2,~] = color_map2(color_vec1, state1, window_size_1, n_component, sel_id);
    s1 = start_region+1;
    s2 = start_region+window_size;
    fprintf('%d %d %d %d\r\n',id1,id2,s1,s2);
    figure;
    set(gcf,'Visible','off');
    imshow(uint8(color1));
    title(sprintf('chr%d iter %d region %d %d',chrom,iter_id,i,window_size));
    saveas(gcf,output_filename1);
    close(gcf);
end


