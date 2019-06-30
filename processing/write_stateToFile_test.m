%% write state to files
function len_vec1 = write_stateToFile_test(state_vec,len_vec_ori,chrom1,...
                                          bin_size,output_path,annotation)
  
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

filename = sprintf('%s/estimate_test%d.%s.txt',output_path,chrom1,annotation);
fid = fopen(filename,'w');

b = len_vec_ori(:,end)==chrom1;
len_vec = len_vec_ori(b,:);

num_region = size(len_vec,1);
bin = bin_size;
% chrom2 start1 stop1 chrom2 start2 stop2 state
format = '%d\t%d\t%d\t%d\t%d\t%d\t%d\r\n';
len_vec1 = zeros(num_region,7);
id_start1 = 1;
for region_id = 1:num_region
    id1 = len_vec(region_id,2)+1;
    id2 = len_vec(region_id,3);
    
    window_size1 = len_vec(region_id,4);
    window_size2 = len_vec(region_id,5);
    
    start_region1 = len_vec(region_id,6);
    start_region2 = len_vec(region_id,7);
    
    a1 = (0:window_size1-1)+start_region1;
    a2 = (0:window_size2-1)+start_region2;
    t2 = repmat(a2,window_size1,1);
    t1 = repmat(a1',window_size2,1);
    t2 = reshape(t2,window_size1*window_size2,1);
    pos = double([t1 t2])*bin;
    
    b1 = id1:id2;
    t_state1 = state_vec(b1);
    sym_type = len_vec(region_id,9);
    if sym_type==0
        t_state1 = reshape(t_state1,window_size1,window_size2);
        pos1 = [pos(:,1) pos(:,1)+bin];
        pos2 = [pos(:,2) pos(:,2)+bin];
    else
        t_state1a = zeros(window_size1,window_size2);
        id1 = index_sym1(window_size1,window_size2);
        t_state1a(id1) = t_state1;
        t_state1a = t_state1a';
        t_state1a(id1) = t_state1;
        t_state1 = t_state1a;
        b = find(pos(:,2)>=pos(:,1));
        pos1 = [pos(b,1) pos(b,1)+bin];
        pos2 = [pos(b,2) pos(b,2)+bin];
    end
    filename = sprintf('%s/estimate_test%d.%d.%s.txt',output_path,chrom1,region_id,annotation);
    dlmwrite(filename,t_state1,'delimiter','\t','precision','%d');
    
    t_state = state_vec(b1);
    
    num2 = length(b1);
    fprintf('%d %d\r\n',region_id,num2);
    id_stop1 = id_start1+num2-1;
    len_vec1(region_id,:) = [num2,id_start1,id_stop1,...
                             window_size1,window_size2,start_region1,start_region2];
    id_start1 = id_stop1+1;
    
    for k = 1:num2
       fprintf(fid,format,chrom1,pos1(k,:),chrom1,pos2(k,:),t_state(k));
    end
end
fclose(fid);

filename = sprintf('%s/test%d.region.txt',output_path,chrom1);
dlmwrite(filename,len_vec1,'delimiter','\t','precision','%d');






