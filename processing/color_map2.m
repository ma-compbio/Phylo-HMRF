%%
function [color1,color2,cnt_vec] = color_map2(color_vec1, state1, window_size,n_component,sel_id)

n_sample = length(state1);

if n_sample~=window_size(1)*window_size(2)
    fprintf('sample size error! %d %d %d',n_sample,window_size);
    return
end

color1 = zeros(n_sample,3);
cnt_vec = zeros(n_component,1);
for i=1:n_component
    b = find(state1==i);
    num1 = length(b);
    cnt_vec(i) = num1;
    if i==sel_id
        t_color = [0,0,0];
    else
        if sel_id==0
            t_color = color_vec1(i,:);
        else
            t_color = color_vec1(1,:);
        end
    end
    color1(b,:) = ones(num1,1)*t_color;
end

color1 = reshape(color1,window_size(1),window_size(2),3);
color2 = reshape(state1,window_size);

end
