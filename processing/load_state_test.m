%% load the states
filename1 = 'estimate_ou_1_4.00_20.mat'; % please specific the filename of state estimation results
data2 = load(filename1);
state_vec = data2.state_vec + 1; % index starting from 1
len_vec = double(data2.len_vec);
iter = data2.iter_id2;
len1 = length(state_vec);
n_component = length(unique(state_vec));
num1 = size(len_vec,1);
region_num = size(len_vec,1);
color_vec1 = importdata('color_vec.txt');

%% feature vectors
% diagonal_state = local_state(feature_mtx,state_vec);
% [m_vec1a,m_vec2a,state_vec_unique] = query_stateFeature1(feature_mtx,state_vec);

%% load regional feature matrix
chrom_id = 1;
chrom_vec = unique(len_vec(:,end));
chrom_num = length(chrom_vec);
bin_size = 50000;
n_iter = 1;
filename_idx = 1;
output_path = 'test1'; % please specify the directory to save the output files
for i = 1:chrom_num
    chrom_id = chrom_vec(i);
    fprintf('chr%d\r\n',chrom_id);
    b = find(len_vec(:,end)==chrom_id);
    region_num = length(b);
    
    %% load regional state vector
    len1 = size(state_vec,1);
    [state_mtx_vec, len_vec1] = read_state_test(state_vec, chrom_id, ...
        len_vec,color_vec1,n_iter,n_component,filename_idx,iter,output_path);
    
    state_vec = write_stateToFile_test(state_vec,len_vec,chrom_id,bin_size,output_path);
    
end





