#encoding:utf-8
#-*- coding: utf-8 -*-  
from optparse import OptionParser
import os.path
import pandas as pd
import numpy as np
import os
import sys
import math
import random
import scipy
import scipy.io

from sklearn import preprocessing
import sklearn.preprocessing
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn import datasets
from scipy.stats import chisquare
from scipy.stats import binom
from scipy.stats import binom_test
import numpy as np
from scipy.misc import logsumexp

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from scipy.special import comb

import warnings
import multiprocessing as mp

from skimage import feature

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral)

import medpy
from medpy.filter.smoothing import anisotropic_diffusion

def near_interpolation1(mtx, window_size):

	window_size = 3
	n1, n2 = mtx.shape[0], mtx.shape[1]
	
	h = int((window_size-1)/2)
	threshold = 1e-05
	cnt1 = 0
	cnt2 = 0
	for i in range(2,n1-1):
		for j in range(i,n2-1):
			if mtx[i,j]<threshold:
				cnt1 += 1
				window = mtx[i-h:i+h+1,j-h:j+h+1]
				window = window.ravel()
				idx = range(0,window_size**2)
				neighbor_idx = np.setdiff1d(idx,int((window_size**2-1)/2))
				neighbor_feature = window[neighbor_idx]
				m1 = np.median(neighbor_feature)
				# m1 = np.median(window)
				if m1>threshold:
					cnt2 += 1
					mtx[i,j] = m1
					mtx[j,i] = m1

	print "cnt1: %d cnt2: %d"%(cnt1,cnt2)

	return mtx

def cnt_estimate(state,n_components):

	cnt_vec = np.zeros(n_components)
	state_vec = np.unique(state)
	num1 = len(state_vec)
	for i in range(0,n_components):
		if i<num1:
			cnt_vec[i] = np.sum(state==state_vec[i])

	# print cnt_vec
	# print cnt_vec/sum(cnt_vec)

	return cnt_vec, cnt_vec/sum(cnt_vec), state_vec

def symmetric_state(state):
	dim1, dim2 = state.shape[0], state.shape[1]
	for i in range(0,dim1):
		for j in range(0,i):
			state[i,j] = state[j,i]

	return state

def symmetric_idx(dim1,dim2):

	a1 = np.ones(dim2).astype(int)
	row_id = np.outer(range(0,dim1),a1)

	a2 = np.ones(dim1).astype(int)
	col_id = np.outer(a2,range(0,dim2))

	row_id = row_id.ravel()
	col_id = col_id.ravel()

	idx = np.where(row_id<=col_id)[0]

	return idx

def meanvalue_state(x,state):

	# data1 = pd.read_table(filename1,header=None)
	vec1 = np.unique(state)
	num1 = len(vec1)	# the number of states

	n_samples, n_features = x.shape[0], x.shape[1]
	
	vec2 = [5,25,50,75,95]
	num2 = len(vec2)
	# stats = dict()
	cnt_vec = np.zeros(num1)
	m_vec1 = []
	for i in range(0,num1):
		print "state %d"%(vec1[i])
		b = np.where(state==vec1[i])[0]
		# m_vec1 = np.zeros((num2,n_features))
		cnt = 0
		for percentile in vec2:
			# m_vec1[cnt] = np.percentile(b, percentile, axis=0)
			print percentile
			temp1 = np.percentile(x[b], percentile, axis=0)
			print temp1
			m_vec1.append(temp1)
			cnt = cnt+1

		# stats[i] = m_vec1
		cnt_vec[i] = len(b)
		
	stats = np.asarray(m_vec1)

	return stats, cnt_vec

# find the indices of serial2 in serial1
# serial1 and serial2 need to be non-negative numbers
def mapping_Idx(serial1,serial2):

	ref_serial = np.sort(serial1)
	ref_sortedIdx = np.argsort(serial1)
	ref_serial = np.int64(ref_serial)
	
	map_serial = np.sort(serial2)
	map_sortedIdx = np.argsort(serial2)
	map_serial = np.int64(map_serial)

	num1 = np.max((ref_serial[-1],map_serial[-1]))+1
	vec1 = np.zeros((num1,2))
	vec1[map_serial,0] = 1
	b = np.where(vec1[ref_serial,0]>0)[0]
	vec1[ref_serial,1] = 1
	b1 = np.where(vec1[map_serial,1]>0)[0]

	idx = ref_sortedIdx[b]
	idx1 = -np.ones(len(map_serial))
	idx1[map_sortedIdx[b1]] = idx

	return np.int64(idx1)
	
# normalize the feature to be in the same scale
# the feature values are non-negative
def normalize_feature(x1,x_min,x_max):

	n1, n2 = x1.shape[0], x1.shape[1]
	vec1 = []

	for i in range(0,n2):
		x = x1[:,i]
		x[x<0] = 0
		m1, m2 = np.min(x), np.max(x)
		vec1.append([m1,m2])
	
	vec1 = np.asarray(vec1)
	min1, max1 = np.median(vec1[:,0]), np.median(vec1[:,1])
	
	if x_min<0:
		x_min = min1
		# x_min = np.exp(min1)-1

	if x_max<0:
		x_max = max1
		# x_max = np.exp(max1)-1

	print x_min, x_max

	for i in range(0,n2):
		x = x1[:,i]
		x[x<0] = 0
		m1, m2 = vec1[i,0], vec1[i,1]
		x1[:,i] = x_min+(x-m1)*1.0*(x_max-x_min)/(m2-m1)

	return x1, vec1, x_min, x_max

# 4-connected or 8-connected neighborhood
def edge_list_grid(data1, serial, window_size, output_filename, num_neighbor):

	# data1 = pd.read_table(filename1)

	num1 = data1.shape[0]	# the number of samples
	# colnames = list(data1)	# chrom, x1, x2, serial, values
	# start_idx = len(colnames)-species_num-3
	# pos = np.asarray(data1.loc[:,colnames[start_idx:start_idx+2]])
	
	# n1, n2 = np.max(pos[:,0]), np.max(pos[:,1])
	# N = np.max((n1,n2))+1	# number of positions
	# print N
	N = window_size

	# serial = N*pos[:,0]+pos[:,1]
	n_neighbor = num_neighbor
		
	if num_neighbor==8:
		neighbor_Mtx = [serial-N-1,serial-N,serial-N+1,serial+1,serial+N+1,serial+N,serial+N-1,serial-1]
	else:
		neighbor_Mtx = [serial-N,serial+1,serial+N,serial-1]

	neighbor_Mtx = np.asarray(neighbor_Mtx).T
	neighbor_Mask = -np.ones(neighbor_Mtx.shape)
	limit1 = N*N

	for i in range(0,n_neighbor):
		b = (neighbor_Mtx[:,i]>=0)&(neighbor_Mtx[:,i]<limit1)
		b1 = np.where(b==True)[0]
		# b2 = np.intersect1d(b1,serial)
		b_1 = mapping_Idx(serial,neighbor_Mtx[b1,i])
		b_2 = np.where(b_1>=0)[0]
		print "neighbor %d %d"%(i,len(b_2))
		id1 = b1[b_2]	# positions with values
		neighbor_Mask[id1,i] = b_1[b_2]

	t_Mtx = np.zeros((num1,n_neighbor,2))
	# neighbors of each of the samples
	# t_Mtx[:,:,0] = np.outer(np.array(range(0,num1)),np.ones(n_neighbor))
	# t_Mtx[:,:,1] = neighbor_Mask
	t_Mtx[:,:,0] = neighbor_Mask
	t_Mtx[:,:,1] = np.outer(np.array(range(0,num1)),np.ones(n_neighbor))

	t_Mtx = t_Mtx.reshape(num1*n_neighbor,2)
	b = np.where(t_Mtx[:,0]>=0)[0]
	edge_list = t_Mtx[b]

	neighbor_Mask1 = neighbor_Mask>=0
	s1 = np.sum(neighbor_Mask1,axis=1)
	b2 = np.where(s1>0)[0]
	
	num2 = len(b2)
	print num2
	cnt_vec = np.zeros(n_neighbor+1)
	
	for i in range(0,n_neighbor+1):
		b = np.where(s1==i)[0]
		cnt_vec[i] = len(b)

	print cnt_vec

	if output_filename!='':
		np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')

	return edge_list

# select part of the data
def select_valuesPosition1(position, x, output_filename, position1, position2, resolution):

	x1, x2 = position[:,0]*resolution, position[:,1]*resolution

	b = (x1>=position1)&(x2<=position2)
	b1 = np.where(b==True)[0]

	# mtx1 = mtx1[b,:]

	if output_filename!="":
		n_fields = 3+x.shape[1]
		colnames = range(0,n_fields)
		data2 = pd.DataFrame(columns=colnames)
		for i in range(0,3):
			data2[colnames[i]] = position[b1,i]

		for i in range(3,n_fields):
			data2[colnames[i]] = x[b1,i-3]
		data2.to_csv(output_filename,index=False,sep='\t')

	return x[b1,:], b1

# write feature vectors to image
def write_matrix_image_Ctrl_v2(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode, filter_param1, filter_param2):

	dim1 = value.shape[-1]
	for i in range(0,dim1):
		temp1 = value[:,i]
		b = np.where(temp1>1e-05)[0]
		print "species %d: %d %.4f"%(i,len(b),np.mean(temp1[b]))
		
	mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)

	for k in range(0,dim1):
		temp1 = mtx1[:,:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		# print "after write_matrix_image_v1"
		# print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1 ori",np.mean(m1,axis=0)

	for k in range(0,dim1):
		temp1 = m1[:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		# print "after reshape"
		# print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	dim1 = mtx1.shape[-1]
	# sigma = 0.5

	window_size = 3
	for i in range(0,dim1):
		temp1 = mtx1[:,:,i]
		b = np.where(temp1<1e-05)[0]
		print "species %d: %d"%(i,len(b))
		mtx1[:,:,i] = near_interpolation1(temp1, window_size)	# use median of neighbors for interpolation 

		temp2 = mtx1[:,:,i]
		b = np.where(temp2<1e-05)[0]
		print "2 species %d: %d"%(i,len(b))

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1",np.mean(m1,axis=0)

	if filter_mode==0:
		
		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			if filter_param1<0:
				mtx1[:,:,i] = anisotropic_diffusion(temp1, niter=10, kappa=50, gamma=0.1, voxelspacing=None, option=1)	# anisotropic diffusion filter
			else:
				mtx1[:,:,i] = anisotropic_diffusion(temp1, niter=filter_param1, kappa=filter_param2, gamma=0.1, voxelspacing=None, option=1)	# anisotropic diffusion filter

	elif filter_mode==1:

		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			if filter_param1<0:
				mtx1[:,:,i] = denoise_bilateral(temp1, sigma_color=0.5, sigma_spatial=5, multichannel=False)	# bilateral filter
			else:
				mtx1[:,:,i] = denoise_bilateral(temp1, sigma_color=filter_param1, sigma_spatial=filter_param2, multichannel=False)	# bilateral filter

	else:
		if sigma>0:
			for i in range(0,dim1):
				temp1 = mtx1[:,:,i]
				mtx1[:,:,i] = scipy.ndimage.filters.gaussian_filter(temp1, sigma)	# Gaussian blur

	output_filename1 = "test1.txt"
	data1, pos_idx, serial = write_matrix_array_v1(mtx1, start_region, output_filename1, type_id)
	
	window_size = mtx1.shape[0]
	# num_neighbor = 8
	# output_filename2 = ""
	edge_list = edge_list_grid(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# write feature vectors to image
# def write_matrix_image_v1(filename, output_filename1, output_filename2)
def write_matrix_image_v1(value, pos, output_filename1):

	# data1 = pd.read_table(filename)
	# num1 = data1.shape[0]	# the number of entries
	# colnames = list(data1)	# chrom, x1, x2, serial, values

	# start_idx = 0
	# if 3 in colnames:
	# 	start_idx = 1
	# print start_idx 

	# pos = np.asarray(data1.loc[:,colnames[start_idx:start_idx+2]])
	# value = np.asarray(data1.loc[:,colnames[start_idx+3:]])

	print "before write to matrix image"
	x1 = value
	dim1,dim2 = x1.shape[0], x1.shape[-1]
	for k in range(0,dim2):
		b2 = np.where(x1[:,k]>1e-05)[0]
		print k, dim1, len(b2), np.mean(x1[:,k]), np.median(x1[:,k]), np.max(x1[:,k]), np.mean(x1[b2,k]), np.median(x1[b2,k])	

	print value.shape

	# n1, n2 = shape1[0], shape1[1]
	start_region1, start_region2 = int(np.min(pos[:,0])), int(np.min(pos[:,1]))
	stop_region1, stop_region2 = int(np.max(pos[:,0])), int(np.max(pos[:,1]))
	
	start_region = np.min((start_region1,start_region2))
	stop_region = np.max((stop_region1,stop_region2))

	window_size = stop_region-start_region+1
	n_samples, dim_feature = value.shape[0], value.shape[1]
	print "n_samples: %d dim_feature: %d window_size: %d start_region: %d stop_region: %d"%(n_samples, dim_feature, window_size, start_region, stop_region)

	# dim_feature = len(colnames)-(start_idx+3)
	# m1_vec = np.zeros((dim_feature,8))
	# value1 = value.copy()
	# for i in range(0,dim_feature):
	# 	temp1 = value[:,i]
	# 	b = np.where(temp1<=0)[0]
	# 	m1, m1a, m1a1, std1 = np.mean(temp1), np.median(temp1), np.max(temp1), np.std(temp1)
	# 	temp1[b] = 1e-10	# default small value
	# 	temp2 = np.log2(temp1)
	# 	m2, m2a, m2a1, std2 = np.mean(temp2), np.median(temp2), np.max(temp2), np.std(temp2)
	# 	m1_vec[i] = [m1,m1a,m1a1,std1,m2,m2a,m2a1,std2]
	# 	value1[:,i] = temp2
	# print m1_vec

	# np.savetxt(output_filename1,m1_vec,fmt='%.2f')
	
	# value = sklearn.preprocessing.StandardScaler().fit_transform(value1)

	# mtx1 = np.zeros(n1,n2,dim_feature)
	mtx1 = np.zeros((window_size,window_size,dim_feature))
	for i in range(0,n_samples):
		id1, id2 = pos[i,0]-start_region, pos[i,1]-start_region		
		mtx1[id1,id2] = value[i,:]
		if id1>id2:
			print "%d error!"%(i)

		# mtx1[id2,id1] = value[i,:]
		if i%100000==0:
			print i, id1, id2, value[i,:]

	if output_filename1!="":
		np.save(output_filename1,mtx1)

	return mtx1, start_region

# write image to feature vectors
def write_matrix_array_v1(mtx1, start_region, output_filename1, type_id):

	# mtx1 = np.zeros((window_size,window_size,dim_feature))
	n1, n2, dim1 = mtx1.shape[0], mtx1.shape[1], mtx1.shape[2]
	
	serial, sel_idx, pos_idx = [], [], []
	cnt = 0
	if type_id==0:		
		for i in range(0,n1):
			for j in range(0,n2):		# the matrix
				# select value
				serial.append(i*n2+j)		
				sel_idx.append(i*n2+j)
				pos_idx.append([i,j])
				cnt = cnt+1
	else:
		for i in range(0,n1):
			for j in range(i,n2):		# the up-triangular matrix
				# select value
				serial.append(i*n2+j)
				sel_idx.append(i*n2+j)
				pos_idx.append([i,j])
				cnt = cnt+1
	
	serial = np.asarray(serial)
	pos_idx = np.asarray(pos_idx)+start_region
	data1 = mtx1.reshape((n1*n2,dim1))
	data_1 = data1[serial,:]
	pos_idx_1 = pos_idx[serial,:]

	if output_filename1!="":
		# np.save(output_filename1,data_1)
		np.savetxt(output_filename1, data_1, delimiter='\t', fmt='%.2f')

	return data_1, pos_idx_1, serial

def multi_contact_matrix(chrom, resolution, filename_list, species, output_filename):

	# species = ['gorGor4','panTro5','panPan2','hg38']	
	print(species)
	species_num = len(species)
	print("species num: ",species_num)
	
	value_dict = dict()
	value_dict1 = dict()
	serial1 = []
	# serial2 = ref_serial.copy()
	t_serial2 = []

	cnt = 0
	for input_path in filename_list:

		filename1 = "%s/chr%s.%dK.txt"%(input_path,chrom,int(resolution/1000))
		if os.path.exists(filename1)==False:
			print "File %s does not exist. Please check."%(filename1)
			return False

		data2 = pd.read_table(filename1,header=None)
		num1 = data2.shape[0]
		x1, x2, value = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2])
		x1, x2 = x1/resolution, x2/resolution

		N1, N2 = x1[-1], x2[-1]
		N = np.max((N1,N2))

		t_vec_serial = np.int64((N+1)*x1 + x2)	# serial of the regions

		b1 = np.where(np.isnan(value)==True)[0]
		value[b1] = -1
		
		num2 = len(b1)
		print num1, num2

		species_id = species[cnt]
		cnt = cnt + 1
		value_dict[species_id] = t_vec_serial	# serial
		# value_dict1[species_id] = value
		value_dict1[species_id] = [x1,x2,value]	# value

		serial1 = np.union1d(serial1,t_vec_serial)
		# serial2 = np.intersect1d(serial2,t_vec_serial)
		t_serial2.append(t_vec_serial)

	serial2 = t_serial2[0]	
	print(len(t_serial2))
	for i in range(1,species_num):
		print(i)
		serial2 = np.intersect1d(serial2,t_serial2[i])

	n1, n2 = len(serial1), len(serial2)
	print "union, intersection", n1, n2

	colnames = [0,1,2]
	colnames.extend(species)
	data_2 = output_multi_contactMtx(serial1, colnames, species, value_dict, value_dict1, resolution, output_filename)

	return list(data_2)

def output_multi_contactMtx(serial1, colnames, species, value_dict, value_dict1, resolution, output_filename1):

	n1 = len(serial1)
	print n1

	data_1 = pd.DataFrame(columns=colnames)

	species_num = len(species)
	mtx1 = np.zeros((n1,species_num))
	mtx_1 = np.zeros((n1,3))
	
	for i in range(0,species_num):
		species_id = species[i]
		temp1 = value_dict[species_id]	# serial
		temp2 = value_dict1[species_id]	# value
		x1, x2, value = temp2[0], temp2[1], temp2[2]
		print len(temp1)
		idx = mapping_Idx(serial1,temp1)
		b1 = np.where(idx>=0)[0]
		if len(b1)>0:
			mtx1[idx[b1],i] = value[b1]
			data_1[species_id] = mtx1[:,i]
			mtx_1[idx[b1],0], mtx_1[idx[b1],1] = x1[b1], x2[b1]

	data_1[0], data_1[1], data_1[2] = np.int64(mtx_1[:,0]), np.int64(mtx_1[:,1]), np.int64(serial1)		

	data_1.to_csv(output_filename1,index=False,sep='\t')

	return data_1

# sourced from hmmlearn
def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum

# sourced from hmmlearn
def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]

# sourced from hmmlearn
def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

