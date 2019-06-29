function sym_Idx = index_sym1(dim1,dim2)

i1 = dim1;
i2 = dim2;
s1 = i1*i2;
temp1 = (1:s1)';
t1 = repmat(1:i2,[i1,1]);
t2 = t1';
t1 = t1(:);
t2 = t2(:);
id1 = t1<=t2;
id_vec1 = temp1(id1);

sym_Idx = id_vec1;

end
