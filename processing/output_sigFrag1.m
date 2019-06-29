%% write signals into seperate file
% write from original imputed data without segment concatenation
function len = output_sigFrag1(chr_annot,chr_loc,sig,id_vec,col_names,col_idx,path_1)
% data1 = importdata(filename1);
% id_vec = importdata(filename2);
% chr_annot = data1.textdata;
% chr_loc = data1.data(:,1:2);
% sig = data1.data(:,3:end);
len = sum(id_vec(:,2));
[n1,n2] = size(sig);
seg_name = cell(n1,1);
for k = 1:n1
    seg_name{k} = num2str(k-1); % index starting from zero
end

%% write signal fragments into BED file
headerlines = {};
col_num = length(col_idx);
num = size(id_vec,1);
bin_size = 6000;
for j = 1:col_num
    col = col_idx(j);
    colname = col_names{col};
    for k = 1:num
        id1 = id_vec(k,1);
        id2 = id_vec(k,1)+id_vec(k,2)-1;
        % fprintf('%d: %d %d\r\n',k,id1,id2);
        if mod(k,1000)==0
            fprintf('%d: %d %d %d\r\n',k,id1,id2,id2-id1+1);
        end
        C = {chr_annot(id1:id2),chr_loc(id1:id2,:),seg_name(id1:id2),sig(id1:id2,col)};
        filename = sprintf('%s/%s_%05d.bed',path_1,colname,k);
        writeToBED(C, filename, headerlines);
    end
end

end