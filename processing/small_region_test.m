%% input: color2 2D matrix
function mtx_1 = small_region_test(chrom,region_id,iter_id,color2,color_vec1,n_component,...
    window_size1,threshold,n_iter,sel_id,output_filename)

[n,m] = size(color2);
window_size = [n,m];

mtx1 = color2(:); % 1d vector of color map
mtx1_copy = mtx1; % copy1
for iter = 1:n_iter
    % sprintf('%d\r\n',iter);
    for state_id = 1:n_component
        mtx2 = mtx1; % copy2
        t_color = reshape(mtx1,n,m);
        
        b1 = mtx1==state_id;
        b2 = mtx1~=state_id;
        mtx2(b1) = 1;
        mtx2(b2) = 0;
        
        mtx2 = reshape(mtx2,n,m);
        
        CC = bwconncomp(mtx2);
        num1 = CC.NumObjects;
        % fprintf('state %d number of objects %d\r\n',state_id,num1);
        
        area_vec = zeros(num1,1);
        pixel = CC.PixelIdxList;
        for j = 1:num1
            temp1 = pixel{j};
            area_vec(j) = length(temp1);
        end
        
        % threshold = 80;
        b = find(area_vec<=threshold);
        num2 = length(b);
        vec1 = zeros(num2,1);
        for k = 1:num2
            id1 = b(k);
            t_pixel = pixel{id1};
            t_state = query_neighbor_state_test(t_pixel, state_id, t_color, ...
                window_size, window_size1, 0.5);
            if t_state~=-1
                vec1(k) = t_state;
                mtx1_copy(t_pixel) = t_state;
            end
        end
    end
    
    mtx1 = mtx1_copy;
    
end

[color1,~,~] = color_map_test(color_vec1,mtx1_copy,window_size,n_component,sel_id);
figure;
set(gcf,'Visible','off');
imshow(uint8(color1));
title(sprintf('chr%d region%d %d %d %d',chrom,region_id,iter_id,window_size(1),window_size(2)));
saveas(gcf,output_filename);
close(gcf);

mtx_1 = mtx1;

end



