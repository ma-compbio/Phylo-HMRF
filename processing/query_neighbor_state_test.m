%% query neighbor states
function neighbor_state = query_neighbor_state_test(pixel_list, state_id, mtx1,...
    window_size, window_size1, ratio_threshold)

num2 = length(pixel_list);
neighbor_state = -1;
vec1 = [];
h = round((window_size1-1)/2);
for i = 1:num2
    value = pixel_list(i);
    i1 = floor(value/window_size(2))+1;
    i2 = mod(value,window_size(2));
    if i1>h && i1+h<=window_size(1) && i2>h && i2+h<=window_size(2)
        window1 = mtx1(i1-h:i1+h,i2-h:i2+h);
        temp1 = window1(:);
        b = temp1~=state_id;
        vec1 = [vec1;temp1(b)];
    end
end
if ~isempty(vec1)
    temp1 = mode(vec1);
    b = sum(vec1==temp1);
    if b>length(vec1)*ratio_threshold
        neighbor_state = temp1;
    else
        neighbor_state = -1;
    end
end

end
