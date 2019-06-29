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

from numpy.lib.recfunctions import append_fields

import sklearn.preprocessing
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn import datasets
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from scipy.special import comb

import copy
import scipy.ndimage

import warnings
import multiprocessing as mp

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral)

import medpy
from medpy.filter.smoothing import anisotropic_diffusion

import multiprocessing as mp
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')

def merge_contact_file(path1, output_filename1):

	# filename1 = 'merge_contact_file.txt'

	chrom_vec = np.asarray(range(1,23))
	colnames1 = ['chrom','start1','start2','value']
	num1 = len(chrom_vec)
	for i in range(0,num1):
		chrom_id = chrom_vec[i]
		print chrom_id
		filename1 = '%s/chr%d.observed.kr.50K.txt'%(path1,chrom_id)
		data1 = pd.read_table(filename1,header=None)
		colnames = list(data1)
		# start1, start2, value = data1[colnames[0]], data1[colnames[1]], data1[colnames[2]]
		t_data = pd.DataFrame(columns=colnames1)
		n1 = data1.shape[0]
		str1 = 'chr%d'%(chrom_id)
		t_data[colnames1[0]] = [str1]*n1
		for j in range(0,3):
			t_data[colnames1[j+1]] = data1[colnames[j]]
		
		if i==0:
			data_1 = t_data
		else:
			data_1 = data_1.append(t_data)

	data_1.to_csv(output_filename1,header=False,index=False,na_rep='NAN',sep='\t')
	
	return True

def merge_estimate_file(path1, species_vec, output_filename1):

	# filename1 = 'merge_contact_file.txt'

	chrom_vec = np.asarray(range(1,23))
	colnames1 = ['chrom','start1','start2','value']
	num1 = len(chrom_vec)
	for i in range(0,num1):
		chrom_id = chrom_vec[i]
		print chrom_id
		filename1 = '%s/test%d.txt'%(path1,chrom_id)
		data1 = pd.read_table(filename1,header=None)
		colnames = list(data1)
		sub_colnames = [colnames[0],colnames[1],colnames[4],colnames[7],colnames[8],colnames[9],colnames[10]]
		# start1, start2, value = data1[colnames[0]], data1[colnames[1]], data1[colnames[2]]
		# t_data = pd.DataFrame(columns=colnames1)
		t_data = data1.loc[:,sub_colnames]
		n1 = t_data.shape[0]
		str1 = 'chr%d'%(chrom_id)
		t_data[colnames[0]] = [str1]*n1
		
		if i==0:
			data_1 = t_data
		else:
			data_1 = data_1.append(t_data)

	data_1.to_csv(output_filename1,header=False,index=False,sep='\t')

	# species_vec = ['hg38','panTro5','panPan2','gorGor4']
	num2 = len(species_vec)
	for i in range(0,num2):
		sub_colnames = [colnames[0],colnames[1],colnames[4],colnames[7+i]]
		data_2 = data_1.loc[:,sub_colnames]
		filename1 = 'estimate_%s.txt'%(species_vec[i])
		data_2.to_csv(filename1,header=False,index=False,sep='\t')

	return True

def intersect_region(file1, file2):

	data1 = pd.read_table(file1,header=None)
	data2 = pd.read_table(file2,header=None)

	chrom1, chrom2 = data1[0], data2[0]
	start1, stop1, start2, stop2 = data1[1], data1[2], data2[1], data2[2]
	serial1, serial2 = data1[3], data2[3]

	num1 = len(serial2)
	matched_Idx = -np.ones(num1)
	t_chrom1, t_start1, t_stop1 = chrom1[serial2], start1[serial2], stop1[serial2]
	
	flag = (t_chrom1==chrom2)&(t_start1<stop2)&(t_stop1>start2)
	b = np.where(flag==True)[0]
	matched_Idx = serial2[b]
	print "matched_Idx", len(matched_Idx), len(matched_Idx)*1.0/len(serial1)
	
	return matched_Idx, serial2

def extend_bed(filename,extension1,output_filename):

	data1 = pd.read_table(filename,header=None)
	chrom, start, stop = data1[0], data1[1], data1[2]
	serial = np.asarray(range(0,len(chrom)))
	
	start = start - extension1
	start[start<0] = 0
	stop = stop + extension1

	# colnames = list(data1)
	colnames = ['chrom','start','stop','serial']
	data2 = pd.DataFrame(columns=colnames)
	data2['chrom'], data2['start'], data2['stop'], data2['serial'] = chrom, start, stop, serial
	data2.to_csv(output_filename,header=False,index=False,sep='\t')

	return True

def write_tobed(filename,output_filename):

	data1 = pd.read_table(filename,header=None)
	chrom, start, stop = data1[0], data1[1], data1[2]
	serial = np.asarray(range(0,len(chrom)))
	
	colnames = ['chrom','start','stop','serial']
	data2 = pd.DataFrame(columns=colnames)
	data2['chrom'], data2['start'], data2['stop'], data2['serial'] = chrom, start, stop, serial
	data2.to_csv(output_filename,header=False,index=False,sep='\t')

	return True

def state_enrichment(chrom1, state_vec, filename1, filename2):

	chrom_vec = np.unique(chrom1)
	state_vec_unique = np.unique(state_vec)
	chrom_num = len(chrom_vec)
	state_num_unique = len(state_vec_unique)
	mtx1 = np.zeros((chrom_num,state_num_unique))
	fold_change = mtx1.copy()

	num1 = len(state_vec)
	ratio_vec = np.zeros(state_num_unique)
	for j in range(0,state_num_unique):
		b = np.where(state1==state_vec_unique[j])
		ratio_vec[j] = len(b)*1.0/num1

	for i in range(0,chrom_num):
		b = np.where(chrom1==chrom_vec[i])
		state1 = state_vec[b]
		t_num1 = length(state1)
		for j in range(0,state_num_unique):
			b = np.where(state1==state_vec_unique[j])
			mtx1[i,j] = len(b)*1.0/t_num1
			fold_change[i,j] = mtx1[i,j]/ratio_vec[j]

	eps = 1e-16
	return np.log2(fold_change+eps), fold_change

def find_region(filename1, threshold):
	
	fid = open(filename1,'r')
	lines = fid.readlines()
	# close the file after reading the lines.
	fid.close()
	num1 = len(lines)
	i = 0
	mdict = {}
	mdict1 = {}
	# threshold = 50*50000
	while i<num1: 
		print i   
		if lines[i][0]!='>' and i+4<num1 and lines[i].find(':')>=0:
			segment = lines[i:i+4]
			t_chrom_vec = []
			t_len_vec = []
			for k in range(0,4):
				if segment[k].find(':')<0:
					print segment[k]
					return False
				vec1 = segment[k].split(' ')
				vec1 = vec1[0].split(':')
				t1 = vec1[0].split('.')
				t2 = vec1[1].split('-')
				t_chrom_vec.append(t1[1])
				start, stop = int(t2[0]), int(t2[1])
				len1 = stop-start
				t_len_vec.append([start,stop,len1])

			print t_chrom_vec, t_len_vec
			t_len_vec1 = np.asarray(t_len_vec)
			flag = find_region1(t_chrom_vec, t_len_vec1[:,-1], threshold)
			print i,flag
			if flag==True:
				chrom = t_chrom_vec[0]
				if chrom in mdict.keys():
					mdict[chrom].extend(t_len_vec)
					mdict1[chrom].append(t_len_vec[0])
				else:
					mdict[chrom] = t_len_vec
					mdict1[chrom] = [t_len_vec[0]]
			i = i+4
		else:
			i+=1

	return mdict, mdict1

def find_region1(chrom_vec, len_vec, threshold):
	num1 = len(chrom_vec)
	chrom = chrom_vec[0]
	for i in range(1,num1):
		if chrom=='chr2':
			if chrom_vec[i]!='chr2A' and chrom_vec[i]!='chr2B' and chrom_vec[i]!='chr2':
				return False
		else:
			if chrom_vec[i]!=chrom:
				return False

	if min(len_vec)<threshold:
		return False

	return True

def cnt_estimate(x, state, n_components):

	# x = np.load(filename1)
	n_features = x.shape[0]
	cnt_vec = np.zeros(n_components)
	state_vec = np.unique(state)
	threshold1 = [0.003,0.25,0.5,0.75,0.997]
	m_vec1 = []
	num1 = len(state_vec)
	for i in range(0,num1):
		state1 = state_vec[i]
		b1 = np.where(state==state1)[0]
		cnt_vec[state1] = len(b1)
		x1 = x[b]

		for j in range(0,n_features):
			t_x1 = x1[:,j]
			temp1 = np.quantile(t_x1,threshold1)
			temp2 = [state1]
			temp2.extend(list(temp1))
			m_vec1.append(temp2)

	return cnt_vec, np.asarray(m_vec1)

def load_data_chromosome2(chrom_vec, x_max, x_min, resolution, num_neighbor, filter_mode, sigma, diagonal_typeId, 
							ref_filename, filename_list, species, data_path, annotation):

	edge_list_vec = []
	samples = []
	# sample_id = []
	len_vec = []
	id1, id2 = 0, 0
	start = time.time()

	queue1 = mp.Queue()
	# chrom_vec = range(20,22)
	# chrom_vec = [1,9,10]

	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
	processes = [mp.Process(target=load_data_chromosome_sub1_2, 
				args=(chrom_id, x_max, x_min, resolution, num_neighbor, filter_mode, sigma, diagonal_typeId, 
					ref_filename, filename_list, species, data_path, queue1)) for chrom_id in chrom_vec]

	# Run processes
	for p in processes:
		p.start()

	results = [queue1.get() for p in processes]
	print(len(results))

	# Exit the completed processes
	print("join")
	for p in processes:
		p.join()

	end = time.time()
	print("use time load chromosomes: %s %s %s"%(start, end, end-start))

	chrom_num = len(chrom_vec)
	chrom_vec1 = np.zeros(chrom_num)
	for i in range(0,chrom_num):
		vec1 = results[i]
		chrom_vec1[i] = vec1[0]

	sort_idx = np.argsort(chrom_vec1)

	samples = []
	len_vec = []
	edge_list_vec = []
	n_samples_accumulate = 0
	id_1 = 0
	id_2 = 0
	for id1 in sort_idx:
		vec1 = results[id1]
		t_samples, t_lenvec, t_edgelistVec = vec1[1], vec1[2], vec1[3]

		samples.extend(t_samples)
		for temp1 in t_lenvec:
			temp1[1] += n_samples_accumulate
			temp1[2] += n_samples_accumulate
			len_vec.append(temp1)
		
		n_samples = t_samples.shape[0]
		n_samples_accumulate += n_samples
		print "chr%s: %d"%(vec1[0],n_samples)
		edge_list_vec.extend(t_edgelistVec)

	return np.asarray(samples), len_vec, edge_list_vec

# load data for diagonal regions and off-diagonal regions
def load_data_chromosome_sub1_2(chrom_id, x_max, x_min, resolution, num_neighbor, filter_mode, 
									sigma, diagonal_typeId, ref_filename, filename_list, species, data_path, m_queue):

	# resolution = 50000
	chrom = str(chrom_id)
	# write contact frequency of different species into an array (n_samples*(n_position+n_species))
	# ref_species = 'hg38'
	type_id = 0
	# output_filename = "%s_test1.txt"%(chrom)
	output_filename = ""	# if want to save the aligned contact files, please specify the filename
	# multi_contact_matrix3A(chrom, resolution, output_filename, type_id)
	data_ori = multi_contact_matrix3A(chrom, resolution, ref_filename, filename_list, species, output_filename, type_id)

	species_num = len(species)
	# filename1 = output_filename
	# data_ori = pd.read_table(filename1)	# load DataFrame file
	colnames = list(data_ori)

	position = np.asarray(data_ori.loc[:,colnames[0:3]])
	x1 = np.asarray(data_ori.loc[:,colnames[3:]])
	print x1.shape
	# print x1[0:10]
	
	x1, vec1, x_min, x_max = normalize_feature(x1,x_min,x_max)
	print np.max(x1,axis=0), np.min(x1,axis=0), np.std(x1,axis=0)
	print vec1

	# log transformation
	x = np.log(1+x1)

	# num_neighbor = 8
	print "num_neighbor: %d"%(num_neighbor)
	type_id = 0

	region_list = []
	# data_path = "inferCars/DATA/chrom"

	# filename3 = "%s/chr%s.synteny.50.hg38.txt"%(data_path,chrom)
	filename3 = "%s/chr%s.synteny.txt"%(data_path,chrom)
	t_lenvec = np.loadtxt(filename3, dtype='int', delimiter='\t')	# load *.txt file
	temp1 = np.ravel(t_lenvec)
		
	if len(temp1)<6:
		region_list.append(t_lenvec)
		region_num = 1
	else:
		region_num = len(t_lenvec)
		for i in range(0,region_num):
			region_list.append(t_lenvec[i])

	print region_list

	# handle the large chromosome size of chr3 and chr6 in genome hg38 
	# this only applies if the reference genome is genome hg38
	region_points_vec = np.asarray([[3,90279522,93797661],[6,57542947,61520508]])
	b1 = np.where(region_points_vec[:,0]==chrom_id)[0]
	region_points = []
	
	if len(b1)>0:
		for id1 in b1:
			region_points.append(region_points_vec[id1,1:])

	print "region_points 1", region_points
	# region_list1: start, stop, length, region_id
	# region_list2: position1,position2,position1,position2,length,length,region_id,region_id1,chrom_id
	region_list1, region_list2_ori = subregion1(filename3, chrom_id, resolution, region_points, type_id)

	num1 = len(region_list2_ori)
	region_list2 = []
	
	if diagonal_typeId==1:
		temp1 = np.asarray(region_list2_ori)
		temp2 = (temp1[:,0]==temp1[:,2])&(temp1[:,1]==temp1[:,3])
		b = np.where(temp2==True)[0]
		region_list2 = [region_list2_ori[idx] for idx in b]
	else:
		region_list2 = region_list2_ori

	print "diagonal_typeId, region_list2", diagonal_typeId, region_list2

	filter_param1, filter_param2 = -1, -1

	start = time.time()
	if filter_mode==0:
		filter_param1, filter_param2 = 5, 50

	param_vec = [resolution, num_neighbor, filter_mode, filter_param1, filter_param2, sigma]

	region_num = len(region_list2)

	queue1 = mp.Queue()
	print("processes")
	start = time.time()
	# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
	processes = [mp.Process(target=load_data_chromosome_sub3, 
				args=(region_id, chrom_id, region_list2, x, position, param_vec, queue1)) for region_id in range(0,region_num)]

	# Run processes
	for p in processes:
		p.start()

	results = [queue1.get() for p in processes]
	print(len(results))

	# Exit the completed processes
	print("join")
	for p in processes:
		p.join()

	region_vec1 = np.zeros(region_num)
	for i in range(0,region_num):
		vec1 = results[i]
		region_vec1[i] = vec1[0]

	sort_idx = np.argsort(region_vec1)

	samples = []
	len_vec = []
	edge_list_vec = []
	id_1 = 0
	id_2 = 0
	for id1 in sort_idx:
		vec1 = results[id1]
		print "region %d"%(vec1[0])
		t_samples, t_lenvec, t_edgelist = vec1[1], vec1[2], vec1[3]
		n_samples = t_samples.shape[0]
		id_2 = id_1 + n_samples
		t_lenvec.insert(1,id_2)
		t_lenvec.insert(1,id_1)
		id_1 = id_2
		print t_lenvec

		samples.extend(t_samples)
		len_vec.append(t_lenvec)
		edge_list_vec.append(t_edgelist)

	m_queue.put((chrom_id, np.asarray(samples), len_vec, edge_list_vec))
	end = time.time()
	print("use time load chromosome %d: %s %s %s"%(chrom_id, start, end, end-start))
		
	return True

def load_data_chromosome_sub3(region_id, chrom_id, region_list, x, position, param_vec, m_queue):

	print "select regions..."

	chrom = str(chrom_id)

	start = time.time()
	t_position1 = region_list[region_id]
	
	position1, position2, position1a, position2a = t_position1[0], t_position1[1], t_position1[2], t_position1[3]
	# region_id, type_id1 = t_position[6], t_position[7]
	region_id1, region_id2 = t_position1[6], t_position1[7]

	resolution, num_neighbor, filter_mode, filter_param1, filter_param2, sigma = param_vec[0], param_vec[1], param_vec[2], param_vec[3], param_vec[4], param_vec[5]

	type_id1 = 0
	if (position1==position1a) and (position2==position2a):
		type_id1 = 1
	
	print position1, position2, position1a, position2a, region_id, type_id1
	# output_filename3 = "%s/chr%s.%dKb.select2.%s.%d.%d.R5.test.1.txt"%(output_path,chrom,int(resolution/1000),annot1,position1,position2)

	output_filename = ""
	# x1, idx = utility1.select_valuesPosition1(position, x, output_filename3, position1, position2, resolution)
	border_type = 0
	x1, idx = select_valuesPosition1_2(position, x, output_filename, position1, position2, position1a, position2a, resolution, border_type)

	t_position = position[idx,:]

	# output_filename1 = "data1_mtx.test.%s.R5.test.txt"%(annot2)
	output_path = "."
	output_filename1 = ""
	output_filename2 = "%s/chr%s.%dKb.edgeList.txt"%(output_path,chrom,int(resolution/1000))

	if(os.path.exists(output_filename1)==True):
		output_filename1 = ""
	if(os.path.exists(output_filename2)==True):
		output_filename2 = ""

	if type_id1==1:
		# x1, mtx1, t_position, edge_list_1 = utility1.write_matrix_image_Ctrl_v2(x1,t_position,output_filename1,output_filename2,
		# 													num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		type_id = 1
		print "write_matrix_image_Ctrl_unsym1"
		x1, mtx1, t_position, edge_list_1 = write_matrix_image_Ctrl_unsym1(x1,t_position,output_filename1,output_filename2,
												num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		start_region1 = np.min(t_position)
		start_region2 = start_region1
	else:
		type_id = 0
		print "write_matrix_image_Ctrl_sym1"
		x1, mtx1, t_position, edge_list_1 = write_matrix_image_Ctrl_sym1(x1,t_position,output_filename1,output_filename2,
															num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		temp1 = np.min(t_position,0)
		start_region1, start_region2 = temp1[0], temp1[1]

	n_samples = x1.shape[0]
	# type_id1=1: diagonal type; type_id1=0: off-diagonal type
	t_lenvec = [n_samples,mtx1.shape[0],mtx1.shape[1],start_region1,start_region2,region_id1,type_id1,chrom_id]
	m_queue.put((region_id, x1, t_lenvec, edge_list_1))

	end = time.time()
	print "select regions use time %s %d: %s"%(chrom, region_id, (end-start))

	return True

def load_data_chromosome_sub3_position(region_id, chrom_id, region_list, x, position, param_vec, m_queue):

	print "select regions..."

	chrom = str(chrom_id)

	start = time.time()
	t_position1 = region_list[region_id]
	
	position1, position2, position1a, position2a = t_position1[0], t_position1[1], t_position1[2], t_position1[3]
	# region_id, type_id1 = t_position[6], t_position[7]
	region_id1, region_id2 = t_position1[6], t_position1[7]

	resolution, num_neighbor, filter_mode, filter_param1, filter_param2, sigma = param_vec[0], param_vec[1], param_vec[2], param_vec[3], param_vec[4], param_vec[5]

	type_id1 = 0
	if (position1==position1a) and (position2==position2a):
		type_id1 = 1
	
	print position1, position2, position1a, position2a, region_id, type_id1
	# output_filename3 = "%s/chr%s.%dKb.select2.%s.%d.%d.R5.test.1.txt"%(output_path,chrom,int(resolution/1000),annot1,position1,position2)

	output_filename = ""
	# x1, idx = utility1.select_valuesPosition1(position, x, output_filename3, position1, position2, resolution)
	border_type = 0
	x1, idx = select_valuesPosition1_2(position, x, output_filename, position1, position2, position1a, position2a, resolution, border_type)

	t_position = position[idx,:]

	output_path = "."
	# output_filename1 = "data1_mtx.test.%s.R5.test.txt"%(annot2)
	output_filename1 = ""
	output_filename2 = "%s/chr%s.%dKb.edgeList.txt"%(output_path,chrom,int(resolution/1000))

	if(os.path.exists(output_filename1)==True):
		output_filename1 = ""
	if(os.path.exists(output_filename2)==True):
		output_filename2 = ""

	if type_id1==1:
		# x1, mtx1, t_position, edge_list_1 = utility1.write_matrix_image_Ctrl_v2(x1,t_position,output_filename1,output_filename2,
		# 													num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		type_id = 1
		print "write_matrix_image_Ctrl_unsym1"
		x1, mtx1, t_position, edge_list_1 = write_matrix_image_Ctrl_unsym1_position(x1,t_position,output_filename1,output_filename2,
												num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		start_region1 = np.min(t_position)
		start_region2 = start_region1
	else:
		type_id = 0
		print "write_matrix_image_Ctrl_sym1"
		x1, mtx1, t_position, edge_list_1 = write_matrix_image_Ctrl_sym1(x1,t_position,output_filename1,output_filename2,
															num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)
		temp1 = np.min(t_position,0)
		start_region1, start_region2 = temp1[0], temp1[1]

	n_samples = x1.shape[0]
	# type_id1=1: diagonal type; type_id1=0: off-diagonal type
	t_lenvec = [n_samples,mtx1.shape[0],mtx1.shape[1],start_region1,start_region2,region_id1,type_id1,chrom_id]
	m_queue.put((region_id, x1, t_lenvec, edge_list_1, t_position))

	end = time.time()
	print "select regions use time %s %d: %s"%(chrom, region_id, (end-start))

	return True

# symmetric matrix
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

# general matrix
def near_interpolation1a(mtx, window_size):

	window_size = 3
	n1, n2 = mtx.shape[0], mtx.shape[1]
	
	h = int((window_size-1)/2)
	threshold = 1e-05
	cnt1 = 0
	cnt2 = 0
	for i in range(2,n1-1):
		for j in range(2,n2-1):
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
					# mtx[j,i] = m1

	print "cnt1: %d cnt2: %d"%(cnt1,cnt2)

	return mtx

def near_interpolation2(mtx, window_size):

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
				m1 = np.median(window)
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

def symmetric_state1(state, window_size):
	
	t_state1 = np.zeros((window_size,window_size))
	id1 = symmetric_idx(window_size,window_size)
	t_state1[id1] = state
	state1 = symmetric_state(t_state1)

	return state1

def symmetric_state1_vec(state_vecList, len_vec):
	
	num1 = len_vec.shape[0]
	state_vecList1 = []
	for i in range(0,num1):
		state1 = symmetric_state1(state_vecList[i])
		state_vecList1.extend(state1)

	return state_vecList1

def symmetric_idx(dim1,dim2):

	a1 = np.ones(dim2).astype(int)
	row_id = np.outer(range(0,dim1),a1)

	a2 = np.ones(dim1).astype(int)
	col_id = np.outer(a2,range(0,dim2))

	row_id = row_id.ravel()
	col_id = col_id.ravel()

	idx = np.where(row_id<=col_id)[0]

	return idx

def symmetric_idx1(dim1,dim2):

	a1 = np.ones(dim2).astype(int)
	row_id = np.outer(range(0,dim1),a1)

	a2 = np.ones(dim1).astype(int)
	col_id = np.outer(a2,range(0,dim2))

	row_id = row_id.ravel()
	col_id = col_id.ravel()

	idx = np.where(row_id<=col_id)[0]
	idx1 = np.where(row_id>=col_id)[0]

	return idx,idx1

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

# compare two sets of labels
def compare_labeling(label1, label2):
	nmi = normalized_mutual_info_score(label1, label2)
	adj_mi = adjusted_mutual_info_score(label1, label2)
	adj_ri = adjusted_rand_score(label1, label2)

	label1_vec, label2_vec = np.unique(label1), np.unique(label2)
	num1, num2 = label1_vec.shape[0], label2_vec.shape[0]
	n1, n2 = label1.shape[0], label2.shape[0]
	pair = np.c_[(label1,label2)]
	tp = 0
	for i in set(label1):
		t1 = np.bincount(pair[pair[:,0]==i,1])
		tp = tp + comb(t1,2).sum()

	a = comb(np.bincount(label2),2).sum()
	b = comb(np.bincount(label1),2).sum()
	
	fp = a-tp   # false positive
	fn = b-tp   # false negative
	s1 = comb(n1,2)
	tn = s1-tp-fp-fn
	ri = (tp+tn)/s1
	precision = tp/a
	recall = tp/b
	f1 = 2*precision*recall/(precision+recall)

	return nmi, adj_mi, adj_ri, ri, precision, recall, f1

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

def normalize_feature2(position,x1,x_min,x_max,norm_typeId=0):

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

	threshold1 = 0.997
	threshold2 = 0.9545
	b1 = np.where(position[:,0]==position[:,1])[0]
	for i in range(0,n2):
		x = x1[:,i]
		x[x<0] = 0
		# m1, m2 = vec1[i,0], vec1[i,1]
		
		b2 = np.where(x[b1]>0)[0]
		diagonal_x = x[b1[b2]]
		print "diagonal_x", len(b1), len(b2)
		if norm_typeId==0:
			limit1 = np.quantile(diagonal_x,threshold1)
		elif norm_typeId==1:
			limit1 = np.quantile(diagonal_x,threshold2)
		elif norm_typeId==2:
			Q1,Q3 = np.quantile(diagonal_x,0.25), np.quantile(diagonal_x,0.75)
			limit1 = Q3+(Q3-Q1)*1.5
		else:
			limit1 = np.max(x)

		b_1 = np.where(x>limit1)[0]
		ratio = len(b_1)*1.0/len(x)
		print "normalize %d %d %.2f %d %.2f"%(norm_typeId,i,limit1,len(b_1),ratio)
		if len(b_1)>0:
			x[b_1] = limit1

		m1, m2 = vec1[i,0], limit1
		x1[:,i] = x_min+(x-m1)*1.0*(x_max-x_min)/(m2-m1)

	return x1, vec1, x_min, x_max

# normalize the feature to be in the same scale
def normalize_feature1(x1,x_min,x_max):

	n1, n2 = x1.shape[0], x1.shape[1]
	vec1 = []
	for i in range(0,n2):
		x = x1[:,i]
		m1, m2 = np.min(x), np.max(x)
		vec1.append([m1,m2])
		x1[:,i] = x_min+(x-m1)*1.0*(x_max-x_min)/(m2-m1)
		
	vec1 = np.asarray(vec1)

	return x1, vec1

def quantileNormalize(df_input):
	df = df_input.copy()
	#compute rank
	dic = {}
	for col in df:
		dic.update({col : sorted(df[col])})
	sorted_df = pd.DataFrame(dic)
	rank = sorted_df.mean(axis = 1).tolist()
	#sort
	for col in df:
		t = np.searchsorted(np.sort(df[col]), df[col])
		df[col] = [rank[i] for i in t]
	return df

def reciprocal_mapping(ref_filename,filename1,filename2,output_filename,annot):

	table1 = pd.read_table(filename1,header=None)
	columns1 = list(table1.columns.values)
	serial1 = table1[columns1[3]]

	table2 = pd.read_table(filename2,header=None)
	columns2 = list(table2.columns.values)
	serial2 = table2[columns2[3]]

	ref_table = pd.read_table(ref_filename,header=None)
	ref_serial = ref_table[3]

	t_serial = serial2
	idx1 = np.int64(mapping_Idx(ref_serial,t_serial))

	cnt = sum(idx1!=serial2)
	print cnt
	if cnt>0:
		print "error!"
	
	t_table = ref_table.loc[idx1,:]
	num1 = t_table.shape[0]
	filename = '%s.subTable.txt'%(annot)
	t_table.to_csv(filename,index=False,sep='\t')
	
	chrom, start, stop = t_table[0], t_table[1], t_table[2]
	chrom2, start2, stop2 = table2[0], table2[1], table2[2]
	b = (chrom2==chrom)&(start2<stop)&(stop2>start)
	b = np.asarray(b)
	b2 = np.where(b==True)[0]
	serial2 = np.asarray(serial2)

	if len(b2)>0:
		t_serial1 = serial2[b2]	# reciprocal mapping indices
	else:
		print "error!"
		return

	filename = '%s.serial.txt'%(annot)
	np.savetxt(filename,t_serial1,fmt='%d')
	idx2 = np.int64(mapping_Idx(np.asarray(serial1),t_serial1))
	filename = '%s.serial1.txt'%(annot)
	temp1 = np.array((idx2,t_serial1)).T
	np.savetxt(filename,temp1,fmt='%d')

	sub_table1 = table1.loc[idx2,:]
	sub_refTable = ref_table.loc[t_serial1,:]

	filename = '%s.subtable.serial1.txt'%(annot)
	sub_table1.to_csv(filename,index=False,sep='\t')

	columns = ['chrom','start','stop','serial','chrom_'+annot,'start_'+annot,'stop_'+annot]
	df1 = pd.DataFrame(columns=columns)
	for i in range(0,4):
		df1[columns[i]] = np.asarray(sub_refTable[i])
	for i in range(0,3):
		df1[columns[i+4]] = np.asarray(sub_table1[i])

	df1.to_csv(output_filename,index=False,sep='\t')

def reciprocal_mapping1(filename_list, annot_list, output_filename):

	idx_list = []
	serial_list = []
	table_list = []
	cnt = 0
	for filename in filename_list:
		data1 = pd.read_table(filename)
		table_list.append(data1)
		serial_list.append(np.asarray(data1['serial']))
		cnt = cnt + 1

	serial1 = serial_list[0]
	for i in range(1,cnt):
		serial1 = np.intersect1d(serial1,serial_list[i])

	for i in range(0,cnt):
		idx = mapping_Idx(serial_list[i],serial1)
		if np.min(idx)<0:
			print "error!", i
		idx_list.append(idx)

	# columns = ['chrom','start','stop','serial','chrom_'+annot1,'start_'+annot1,'stop_'+annot1]
	df1 = table_list[0].loc[idx_list[0],:]
	for i in range(1,cnt):
		annot1 = annot_list[i]
		print annot1
		t_columns = ['chrom_'+annot1,'start_'+annot1,'stop_'+annot1]
		for k in range(0,3):
			df1[t_columns[k]] = np.asarray(table_list[i].loc[idx_list[i],t_columns[k]])
		# df1[t_columns[1:3]] = df1[t_columns[1:3]].astype(int)
	
	# df1 = df1.infer_objects()

	df1.to_csv(output_filename,index=False,sep='\t')

	return df1

def concatenate_1(filename1,filename2):
	#idlist = np.loadtxt(filename2,dtype={'names':('start','len','interval'),
	#									'formats':('int32','int32','int32')})
	with open(filename1) as f:
		filenames = f.readlines()
	num = len(filenames)
	print "filenames", num

	fid = open(filename2,'w+')
	num_vec = []
	for filename in filenames:
		filename = filename.rstrip('\n')
		print filename
		with open(filename) as f:
			lines = f.readlines()
			num_vec.append(len(lines))
			fid.writelines(lines)
	
	print num_vec
	return num_vec

# input:
# filename1: estimated states
# filename2: color file
# sel_idx: selected column names
# output_filename: output file
def write_toRGB(filename1, filename2, output_filename):
	
	data2 = pd.read_table(filename1,header=None)

	colnames = list(data2)
	pos1, pos2 = np.asarray(data2[colnames[0]]), np.asarray(data2[colnames[1]])
	state1 = np.asarray(data2[colnames[-1]])
	print state1[0:10]

	start_region = np.min((np.min(pos1),np.min(pos2)))
	stop_region = np.max((np.max(pos1),np.max(pos2)))

	window_size = stop_region-start_region+1

	print start_region, stop_region, window_size

	n_components = max(state1)+1
	state_id = range(1,n_components+1)	# original states

	n_sample = len(state1)
	print n_sample

	data3 = pd.read_table(filename2,header=None)
	colnames1 = list(data3)
	t_color = data3[colnames1[-1]]
	color_vec = list(set(t_color))
	print color_vec

	color1 = [None]*n_sample

	state2 = np.int64(np.zeros(n_sample))
	color1 = np.zeros((window_size,window_size,3))
	color2 = np.zeros((window_size,window_size))
	
	cnt_vec = np.zeros(n_components)
	for i in range(0,n_components):
		b = np.where(state1==i)[0]
		cnt_vec[i] = len(b)

	print cnt_vec

	for i in range(0,n_sample):
		t_state = state1[i]
		state2[i] = state_id[t_state]
		temp1 = color_vec[state2[i]-1]
		temp2 = temp1.split(',')
		id1 = pos1[i]-start_region
		id2 = pos2[i]-start_region
		for k in range(0,3):
			color1[id1,id2,k] = int(temp2[k])
		color2[id1,id2] = state2[i]

	print color1.shape

	state_rgb = dict()
	state_rgb['state1'] = color1
	state_rgb['state2'] = color2
	scipy.io.savemat(output_filename,state_rgb)

	return True

def edge_list_grid(filename1, output_filename):

	data1 = pd.read_table(filename1)

	num1 = data1.shape[0]	# the number of entries
	colnames = list(data1)	# chrom, x1, x2, serial, values
	pos = np.asarray(data1.loc[:,colnames[1:3]])
	
	serial = np.zeros(num1)
	n1, n2 = np.max(pos[:,0]), np.max(pos[:,1])
	N = np.max((n1,n2))+1	# number of positions
	print N

	for i in range(0,num1):
		x1, x2 = pos[i,0], pos[i,1]
		serial[i] = N*x1+x2

	edge_list = []
	print num1
	for i in range(0,num1):
		if i%1000==0:
			print i
		t_serial = serial[i]
		t_neighbor = [t_serial-N,t_serial+1,t_serial+N,t_serial-1]
		t_neighbor = np.asarray(t_neighbor)
		for s1 in t_neighbor:
			b1 = np.where(serial==s1)[0]
			if len(b1)>0:
				edge_list.append([i,b1[0]])

	edge_list = np.asarray(edge_list)

	if output_filename!='':
		np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')

	return edge_list

# sort array by first column and then by second column
def _sort_array(array1):
		
	sorted_array = np.asarray(sorted(array1,key=lambda x:(x[0],x[1])))

	return sorted_array

# 4-connected or 8-connected neighborhood
def edge_list_grid3_sym(data1, serial, window_size, output_filename, num_neighbor):

	# data1 = pd.read_table(filename1)

	num1 = data1.shape[0]	# the number of samples
	# colnames = list(data1)	# chrom, x1, x2, serial, values
	# start_idx = len(colnames)-species_num-3
	# pos = np.asarray(data1.loc[:,colnames[start_idx:start_idx+2]])
	
	# n1, n2 = np.max(pos[:,0]), np.max(pos[:,1])
	# N = np.max((n1,n2))+1	# number of positions
	# print N
	N = window_size
	print N

	# serial = N*pos[:,0]+pos[:,1]
	n_neighbor = num_neighbor

	x = np.int64(serial/N)	# row index
	y = serial%N  # column index

	neighbor_Idx = dict()
	if num_neighbor==8:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y+1))  # lower right
		neighbor_Idx[2] = np.asarray((x+1, y))  # lower
		neighbor_Idx[3] = np.asarray((x+1, y-1))  # lower left
	else:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y))  # lower

	edge_list = []
	num_neighbor1 = int(num_neighbor/2)
	cnt_vec = np.zeros(num_neighbor1)
	print num_neighbor1
	for i in range(0,num_neighbor1):
		temp1 = neighbor_Idx[i]
		print temp1[:,0:100]
		# b1 = np.where(temp1[0]<=temp1[1])[0]
		temp2 = (temp1[0]>=0)&(temp1[0]<N)&(temp1[1]>=0)&(temp1[1]<N)
		b2 = np.where(temp2==True)[0]
		# id1 = np.intersect1d(b1,b2)
		print "sym neighbor%d: %d"%(i,len(b2))
		
		serial1 = temp1[0,b2]*N + temp1[1,b2]
		print serial1[0:100]
		
		b_1 = mapping_Idx(serial,serial1)
		b_2 = np.where(b_1>=0)[0]
		id1 = b_1[b_2]
		#print b_1[0:100]
		
		temp3 = np.asarray((serial[b_2],serial1[b_2])).T
		edge_list.extend(temp3)
		#print temp3[0:100]

		temp3 = np.asarray((serial1[b_2],serial[b_2])).T
		edge_list.extend(temp3)
		#print temp3[0:100]

		cnt_vec[i] = len(b_2)
	
	edge_list = np.asarray(edge_list)
	edge_list = _sort_array(edge_list)

	if output_filename!='':
		np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')

	return edge_list

def _edge_weight_undirected(self, X, edge_list_1, N):

	print "edge weight undirected"
		
	# n_samples = self.n_samples
	# n_edges = len(self.edge_list_1)
	# edge_list_1 = self.edge_list_1
	beta1 = self.beta1
	n_samples = len(X)
	n_edges = len(edge_list_1)
	#beta1 = 0.1 # setting beta1 
	edge_weightList = np.zeros(n_edges)

	if n_edges%2!=0:
		print "number of edges error! %d"%(n_edges)
		
	n_edges1 = int(n_edges/2)
	# edge_weightList_undirected = np.ones(n_edges1).astype(np.float64)
	# edge_idList_undirected = np.zeros((n_edges1,2)).astype(np.int64)	

	eps = 1e-16 
	X_norm = np.sqrt(np.sum(X*X,axis=1))
	cnt = 0
	y_1, y_2 = edge_list_1[:,0]%(N+1), edge_list_1[:,1]%(N+1)
		
	for k in range(0,n_edges):
		j, i = edge_list_1[k,0], edge_list_1[k,1]
		x1, x2 = X[j], X[i]
		difference = np.dot(x1-x2,(x1-x2).T)
		temp1 = difference/(X_norm[i]*X_norm[j]+small_eps)
			
		t_y1, t_y2 = y_1[k], y_2[k]
		if (t_y1==0 and t_y2!=0) or (t_y1!=0 and t_y2==0):
			edge_weightList[k] = 2*np.exp(-beta1*temp1)
		else:
			edge_weightList[k] = np.exp(-beta1*temp1)

	b = np.where(edge_list_1[:,0]<edge_list_1[:,1])[0]
	print "id1<id2:%d"%(len(b))

	print "edge_list_1[b], edge_weightList[b]", edge_list_1[b].shape, edge_weightList[b].shape
	edge_weightList_undirected = np.hstack((edge_list_1[b],edge_weightList[b][:,np.newaxis]))

	#self.edge_weightList = edge_weightList
	# print edge_weightList.shape
	print edge_weightList.shape, edge_weightList_undirected.shape

	position1 = edge_list_1[0,0]
	filename = 'edge_weightList%d.txt'%(position1)
	#np.savetxt(filename, edge_weightList, fmt='%.4f', delimiter='\t')

	filename = 'edge_weightList_undirected%d.txt'%(position1)
	fields = ['start','stop','weight']
	data1 = pd.DataFrame(columns=fields)
	# data1['start'], data1['stop'] = edge_idList_undirected[:,0], edge_idList_undirected[:,1]
	# data1['weight'] = edge_weightList_undirected
	data1['start'], data1['stop'] = edge_weightList_undirected[:,0], edge_weightList_undirected[:,1]
	data1['weight'] = edge_weightList_undirected[:,2]
	data1.to_csv(filename,header=False,index=False,sep='\t')

	print "edge weight output to file"

	# test 
	# edge_weightList_undirected = np.ones(n_edges1).astype(np.float64)
	# edge_weightList = np.ones(n_edges).astype(np.float64)

	# return edge_weightList, edge_idList_undirected, edge_weightList_undirected
	return edge_weightList_undirected

# select part of the data
def select_values(filename1, output_filename, species_num):

	# data1 = pd.read_table(filename1,header=None)
	data1 = pd.read_table(filename1)

	colnames = list(data1)

	# start_idx = 4
	start_idx = len(colnames)-species_num
	mtx1 = data1.loc[:,colnames[start_idx:]]
	mtx1 = np.asarray(mtx1)
	mtx1[mtx1<=0] = 0
	a2 = np.sum(mtx1,axis=1)

	b = np.where(a2>0)[0]
	# mtx1 = mtx1[b,:]
	data2 = data1.loc[b,colnames]

	if output_filename!="":
		data2.to_csv(output_filename,index=False,sep='\t')

	return data2

# select part of the data
def select_valuesPosition(filename1, output_filename, start_idx, position1, position2, resolution):

	data1 = pd.read_table(filename1,header=None)
	colnames = list(data1)

	# start_idx = 4
	mtx1 = data1.loc[:,colnames[start_idx:]]
	a1 = np.asarray(mtx1)
	a2 = np.sum(mtx1,axis=1)

	x1, x2 = np.asarray(mtx1[colnames[start_idx-2]]), np.asarray(mtx1[colnames[start_idx-1]])
	x1, x2 = x1*resolution, x2*resolution

	b = (x1>position1)&(x2<position2)

	b1 = np.where(b==True)[0]

	b = np.where(a2>=0)[0]
	# mtx1 = mtx1[b,:]
	data2 = data1.loc[b,colnames]

	# data2.to_csv(output_filename,header=False,index=False,sep='\t')

	return data2

# select part of the data
def select_valuesPosition1(position, x, output_filename, position1, position2, resolution):

	# x1, x2 = position[:,0]*resolution, position[:,1]*resolution
	x1, x2 = position[:,0]*resolution, (position[:,1]+1)*resolution

	b = (x1>=position1)&(x2<=position2)
	# b = (x1>position1)&(x2<=position2-resolution)
	b1 = np.where(b==True)[0]

	# mtx1 = mtx1[b,:]
	thresh = 0
	num1 = x.shape[1]
	cnt_vec = np.zeros(num1)
	for i in range(0,num1):
		b2 = np.where(x[b1,i]>thresh)[0]
		cnt_vec[i] = len(b2)
	print "species cnt_vec",cnt_vec

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

# select part of the data
def select_valuesPosition1_1(position, x, output_filename, position1, position2, resolution):

	x1, x2 = position[:,0]*resolution, (position[:,1])*resolution

	b = (x1>=position1)&(x2<position2)
	b1 = np.where(b==True)[0]

	thresh = 0
	num1 = x.shape[1]
	cnt_vec = np.zeros(num1)
	for i in range(0,num1):
		b2 = np.where(x[b1,i]>thresh)[0]
		cnt_vec[i] = len(b2)
	print "species cnt_vec",cnt_vec

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

# select part of the data
def select_valuesPosition1_2(position, x, output_filename, position1, position2, position1a, position2a, resolution, border_type=0):

	if border_type==0:
		x1, x2 = position[:,0]*resolution, (position[:,1]+1)*resolution
		b = (x1>=position1)&(x1<=position2)&(x2>=position1a)&(x2<=position2a)
	elif border_type==1:
		x1, x2 = position[:,0]*resolution, (position[:,1]+1)*resolution
		b = (x1>=position1)&(x2<=position2)
	else:
		x1, x2 = position[:,0]*resolution, (position[:,1])*resolution
		b = (x1>=position1)&(x1<position2)&(x2>=position1a)&(x2<position2a) 

	b1 = np.where(b==True)[0]
	thresh = 0

	thresh = 0
	num1 = x.shape[1]
	cnt_vec = np.zeros(num1)

	for i in range(0,num1):
		b2 = np.where(x[b1,i]>thresh)[0]
		cnt_vec[i] = len(b2)
	print "species cnt_vec",cnt_vec

	num1 = x.shape[1]
	cnt_vec = np.zeros(num1)
	for i in range(0,num1):
		b2 = np.where(x[b1,i]>thresh)[0]
		cnt_vec[i] = len(b2)
	print "species cnt_vec",cnt_vec

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

def degree_vertex(filename1, output_filename):

	data1 = pd.read_table(filename1,header=None)
	colnames = list(data1)

	data1 = pd.read_table(filename1)

	num1 = data1.shape[0]	# the number of entries
	colnames = list(data1)	# chrom, x1, x2, serial, values
	pos = np.asarray(data1.loc[:,colnames[1:3]])
	
	serial = np.zeros(num1)
	n1, n2 = np.max(pos[:,0]), np.max(pos[:,1])
	N = np.max((n1,n2))+1	# number of positions
	print N

	for i in range(0,num1):
		x1, x2 = pos[i,0], pos[i,1]
		serial[i] = size1*x1+x2

	edge_list = []
	for i in range(0,num1):
		t_serial = serial[i]
		t_neighbor = [t_serial-N,t_serial+1,t_serial+N,t_serial-1]
		t_neighbor = np.asarray(t_neighbor)
		for s1 in t_neighbor:
			b1 = np.where(serial==s1)[0]
			if len(b1)>0:
				edge_list.append([i,b1[0]])

	edge_list = np.asarray(edge_list)

	if output_filename!='':
		np.savetxt(output_filename,edge_list,fmt='%d')

	return True

# write feature vectors to image
# def write_matrix_image_Ctrl(filename, output_filename1, output_filename2):
def write_matrix_image_Ctrl(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id):

	mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)

	dim1 = mtx1.shape[-1]

	if sigma>0:
		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			mtx1[:,:,i] = scipy.ndimage.filters.gaussian_filter(temp1, sigma)	# Gaussian blur

	output_filename1 = "test1.txt"
	data1, pos_idx, serial = write_matrix_array_v1(mtx1, start_region, output_filename1, type_id)
	
	window_size = mtx1.shape[0]
	# num_neighbor = 8
	# output_filename2 = ""
	edge_list = edge_list_grid3(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# write feature vectors to image
# def write_matrix_image_Ctrl(filename, output_filename1, output_filename2):
def write_matrix_image_Ctrl_v1(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode):

	mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)

	dim1 = mtx1.shape[-1]
	# sigma = 0.5

	window_size = 3
	for i in range(0,dim1):
		temp1 = mtx1[:,:,i]
		mtx1[:,:,i] = near_interpolation2(temp1, window_size)	# use median of neighbors for interpolation 

	if filter_mode==0:
		
		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			mtx1[:,:,i] = anisotropic_diffusion(temp1, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)	# anisotropic diffusion filter

	elif filter_mode==1:

		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			mtx1[:,:,i] = denoise_bilateral(temp1, sigma_color=0.05, sigma_spatial=5, multichannel=False)	# bilateral filter

	elif filter_mode==2:

		for i in range(0,dim1):
			temp1 = mtx1[:,:,i]
			mtx1[:,:,i] = denoise_bilateral(temp1, sigma_color=0.05, sigma_spatial=10, multichannel=False)	# bilateral filter

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
	edge_list = edge_list_grid3(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# write feature vectors to image
# def write_matrix_image_Ctrl(filename, output_filename1, output_filename2):
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
		print "after write_matrix_image_v1"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1 ori",np.mean(m1,axis=0)

	for k in range(0,dim1):
		temp1 = m1[:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after reshape"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

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
	edge_list = edge_list_grid3(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# write feature vectors to unsymmetric image
# compute edge weight
def write_matrix_image_Ctrl_unsym1(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode, filter_param1, filter_param2):

	dim1 = value.shape[-1] 

	for i in range(0,dim1):
		temp1 = value[:,i]
		b = np.where(temp1>1e-05)[0]
		print "species %d: %d %.4f"%(i,len(b),np.mean(temp1[b]))
		
	# write value to matrix
	mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)

	for k in range(0,dim1):
		temp1 = mtx1[:,:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after write_matrix_image_v1"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1 ori",np.mean(m1,axis=0)

	for k in range(0,dim1):
		temp1 = m1[:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after reshape"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	dim1 = mtx1.shape[-1]
	# sigma = 0.5

	# nearest neighbor interpolation
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

	# write matrix to array
	output_filename1 = "test1.txt"
	type_id = 1
	data1, pos_idx, serial = write_matrix_array_v1(mtx1, start_region, output_filename1, type_id)
	
	window_size = mtx1.shape[0]
	edge_list = edge_weightlist_grid3_undirected_unsym(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# write feature vectors to unsymmetric image
# compute edge weight
# use masking matrix
def write_matrix_image_Ctrl_unsym1_position(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode, filter_param1, filter_param2):

	dim1 = value.shape[-1] 

	for i in range(0,dim1):
		temp1 = value[:,i]
		b = np.where(temp1>1e-05)[0]
		print "species %d: %d %.4f"%(i,len(b),np.mean(temp1[b]))
		
	# write value to matrix
	# mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)
	mtx1, start_region, value_index1, value_index2 = write_matrix_image_v1_mask(value, pos, output_filename1)

	for k in range(0,dim1):
		temp1 = mtx1[:,:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after write_matrix_image_v1"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1 ori",np.mean(m1,axis=0)

	for k in range(0,dim1):
		temp1 = m1[:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after reshape"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	dim1 = mtx1.shape[-1]
	# sigma = 0.5

	# nearest neighbor interpolation
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

	# write matrix to array
	output_filename1 = "test1.txt"
	type_id = 1
	data1, pos_idx, serial = write_matrix_array_v1(mtx1, start_region, output_filename1, type_id)

	serial1 = np.intersect1d(serial, value_index1)
	serial2 = np.intersect1d(serial, value_index2)

	b1 = mapping_Idx(serial,serial1)
	serial_1 = serial[b1]

	b2 = mapping_Idx(serial,serial2)
	serial_2 = serial[b2]

	t_data1 = data1[b2,:]
	t_serial = serial_2

	print "value_index1, value_index2", len(value_index1), len(value_index2), len(serial1), len(serial2), len(serial_1), len(serial_2)
	
	window_size = mtx1.shape[0]
	# num_neighbor = 8
	# output_filename2 = ""
	# edge_list = edge_list_grid3_unsym(data1, serial, window_size, output_filename2, num_neighbor)
	edge_list = edge_weightlist_grid3_undirected_unsym(t_data1, t_serial, window_size, output_filename2, num_neighbor)

	return t_data1, mtx1, pos_idx, edge_list

# write feature vectors to symmetric image
# compute edge weight
def write_matrix_image_Ctrl_sym1(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode, filter_param1, filter_param2):

	dim1 = value.shape[-1]
	for i in range(0,dim1):
		temp1 = value[:,i]
		b = np.where(temp1>1e-05)[0]
		print "species %d: %d %.4f"%(i,len(b),np.mean(temp1[b]))
		
	# mtx1, start_region = write_matrix_image_v1(value, pos, output_filename1)
	mtx1, start_region1, start_region2 = write_matrix_image_v1a(value, pos, output_filename1)

	for k in range(0,dim1):
		temp1 = mtx1[:,:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after write_matrix_image_v1"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	m1 = mtx1.reshape((mtx1.shape[0]*mtx1.shape[1],dim1))
	print "m1 ori",np.mean(m1,axis=0)

	for k in range(0,dim1):
		temp1 = m1[:,k]
		x1 = np.ravel(temp1)
		b2 = np.where(x1>1e-05)[0]
		print "after reshape"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	dim1 = mtx1.shape[-1]
	# sigma = 0.5

	window_size = 3
	for i in range(0,dim1):
		temp1 = mtx1[:,:,i]
		b = np.where(temp1<1e-05)[0]
		print "species %d: %d"%(i,len(b))
		mtx1[:,:,i] = near_interpolation1a(temp1, window_size)	# use median of neighbors for interpolation 

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
	type_id = 0
	data1, pos_idx, serial = write_matrix_array_v1a(mtx1, start_region1, start_region2, output_filename1, type_id)

	# num_neighbor = 8
	# output_filename2 = ""
	# edge_list = edge_list_grid3_unsym(data1, serial, window_size, output_filename2, num_neighbor)
	window_size = mtx1.shape
	edge_list = edge_weightlist_grid3_undirected(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

def write_matrix_image_Ctrl_v2_unprocess(value, pos, output_filename1, output_filename2, num_neighbor, sigma, type_id, filter_mode, filter_param1, filter_param2):

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
		print "after write_matrix_image_v1"
		print k, len(x1), len(b2), np.mean(x1), np.median(x1), np.max(x1), np.mean(x1[b2]), np.median(x1[b2])	

	output_filename1 = "test1.txt"
	data1, pos_idx, serial = write_matrix_array_v1(mtx1, start_region, output_filename1, type_id)
	
	window_size = mtx1.shape[0]
	# num_neighbor = 8
	# output_filename2 = ""
	edge_list = edge_list_grid3(data1, serial, window_size, output_filename2, num_neighbor)

	return data1, mtx1, pos_idx, edge_list

# 4-connected or 8-connected neighborhood
def edge_list_grid3(data1, serial, window_size, output_filename, num_neighbor):

	num1 = data1.shape[0]	# the number of samples
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
	# edge_list = _sort_array(edge_list)

	if output_filename!='':
		np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')

	return edge_list

# 4-connected or 8-connected neighborhood
# keep one edge for each pair of nodes
# keep edges for unsymmetric matrix	
def edge_weightlist_grid3_undirected_unsym(data1, serial, window_size, output_filename, num_neighbor):

	num1 = data1.shape[0]	# the number of samples

	# print N
	N = window_size
	print "window_size", N

	# serial = N*pos[:,0]+pos[:,1]
	n_neighbor = num_neighbor

	x = np.int64(serial/N)	# row index
	y = serial%N  # column index

	print x,y
	print serial[0:100]

	t_idx = np.asarray(range(0,N*N))
	m_idx = mapping_Idx(t_idx,serial)
	t_idx1 = -np.int64(np.ones(N*N))
	t_serial = np.asarray(range(0,len(serial)))
	t_idx1[m_idx] = t_serial

	diagonal_idx = np.where(x==y)[0]
	print "diagonal_idx: %d"%(len(diagonal_idx))

	neighbor_Idx = dict()
	if num_neighbor==8:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y+1))  # lower right
		neighbor_Idx[2] = np.asarray((x+1, y))  # lower
		neighbor_Idx[3] = np.asarray((x+1, y-1))  # lower left
	else:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y))  # lower

	edge_list = []
	edge_weightList = []
	num_neighbor1 = int(num_neighbor/2)
	cnt_vec = np.zeros(num_neighbor1)
	print num_neighbor1
	X_norm = np.sqrt(np.sum(data1*data1,axis=1))

	small_eps = 1e-16
	for i in range(0,num_neighbor1):
		temp1 = neighbor_Idx[i]
		b1 = np.where(temp1[0]<=temp1[1])[0]
		temp2 = (temp1[0]>=0)&(temp1[1]<N)
		b2 = np.where(temp2==True)[0]
		id_1 = np.intersect1d(b1,b2)
		serial1 = temp1[0,id_1]*N + temp1[1,id_1]
		
		print serial.shape, serial1.shape
		
		b_1 = mapping_Idx(serial,serial1)
		b_2 = np.where(b_1>=0)[0]

		print "serial1, b_2", serial1.shape, b_2.shape
		
		id1 = id_1[b_2]
		id2 = b_1[b_2]

		print id1, id2
		x1, x2 = data1[id1], data1[id2]
		d1 = np.sum((x1-x2)**2,1)
		print "x1", x1[0:10]
		print "x2", x2[0:10]
		print "d1", d1[0:10]
		weight = d1/(X_norm[id1]*X_norm[id2]+small_eps)
		# weight = np.exp(-beta1*weight)

		b1a = np.where(temp1[0]==temp1[1])[0]
		b1a = np.intersect1d(diagonal_idx,b1a)
		print "b1a %d"%(len(b1a))
		
		if len(b1a)>0:
			serial1a = temp1[0,b1a]*N + temp1[1,b1a]
			print "serial1a", serial1a
			b_1a = mapping_Idx(serial1[b_2],serial1a)
			b_2a = np.where(b_1a>=0)[0]
			id3 = b_1a[b_2a]
			print "id3: %d"%(len(id3))
			weight[id3] = 0.5*weight[id3] # edge weight between diagonal nodes

		temp3 = np.hstack((np.asarray((id1,id2)).T,weight[:, np.newaxis]))
		edge_list.extend(temp3)
		cnt_vec[i] = len(b_2)

	edge_list = np.asarray(edge_list)
	edge_list = _sort_array(edge_list)

	print "edge_list shape", edge_list.shape

	if output_filename!='':
		# np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')
		colnames = ['id1','id2','weight']
		# data2 = pd.DataFrame(columns=colnames,data=edge_list)
		data2 = pd.DataFrame(columns=colnames)
		data2['id1'], data2['id2'] = np.int64(edge_list[:,0]), np.int64(edge_list[:,1])
		data2['weight'] = edge_list[:,2]
		data2.to_csv(output_filename,index=False,header=False,sep='\t')

	return edge_list

def edge_weightlist_grid3_undirected(data1, serial, window_size, output_filename, num_neighbor):

	num1 = data1.shape[0]	# the number of samples
	# beta1 = 0.1

	N1, N2 = window_size[0], window_size[1]

	# serial = N*pos[:,0]+pos[:,1]
	n_neighbor = num_neighbor

	x = np.int64(serial/N2)	# row index
	y = serial%N2  # column index

	diagonal_idx = np.where(x==y)[0]
	print "diagonal_idx: %d"%(len(diagonal_idx))

	neighbor_Idx = dict()
	if num_neighbor==8:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y+1))  # lower right
		neighbor_Idx[2] = np.asarray((x+1, y))  # lower
		neighbor_Idx[3] = np.asarray((x+1, y-1))  # lower left
	else:
		neighbor_Idx[0] = np.asarray((x, y+1))  # right
		neighbor_Idx[1] = np.asarray((x+1, y))  # lower

	edge_list = []
	edge_weightList = []
	num_neighbor1 = int(num_neighbor/2)
	cnt_vec = np.zeros(num_neighbor1)
	print num_neighbor1
	X_norm = np.sqrt(np.sum(data1*data1,axis=1))

	small_eps = 1e-16
	for i in range(0,num_neighbor1):
		temp1 = neighbor_Idx[i]
		temp2 = (temp1[0]>=0)&(temp1[0]<N1)&(temp1[1]>=0)&(temp1[1]<N2)
		b2 = np.where(temp2==True)[0]
		id_1 = b2
		serial1 = temp1[0,id_1]*N2 + temp1[1,id_1]
		
		print serial.shape, serial1.shape
		
		b_1 = mapping_Idx(serial,serial1)
		b_2 = np.where(b_1>=0)[0]

		print "serial1, b_2", serial1.shape, b_2.shape
		
		id1 = id_1[b_2]	# directed to
		id2 = b_1[b_2]	# directed from

		print id1, id2
		x1, x2 = data1[id1], data1[id2]
		d1 = np.sum((x1-x2)**2,1)
		print "x1", x1[0:10]
		print "x2", x2[0:10]
		print "d1", d1[0:10]
		weight = d1/(X_norm[id1]*X_norm[id2]+small_eps)
		# weight = np.exp(-beta1*weight)

		temp3 = np.hstack((np.asarray((id1,id2)).T,weight[:, np.newaxis]))
		edge_list.extend(temp3)
		cnt_vec[i] = len(b_2)

	edge_list = np.asarray(edge_list)
	edge_list = _sort_array(edge_list)

	print "edge_list shape", edge_list.shape

	if output_filename!='':
		# np.savetxt(output_filename,edge_list,fmt='%d',delimiter='\t')
		colnames = ['id1','id2','weight']
		# data2 = pd.DataFrame(columns=colnames,data=edge_list)
		data2 = pd.DataFrame(columns=colnames)
		data2['id1'], data2['id2'] = np.int64(edge_list[:,0]), np.int64(edge_list[:,1])
		data2['weight'] = edge_list[:,2]
		data2.to_csv(output_filename,index=False,header=False,sep='\t')

	return edge_list

# subregion without overlapping
def subregion(filename, resolution, threshold1, threshold2, region_size):

	t_lenvec = np.loadtxt(filename, dtype='int', delimiter='\t')	# load *.txt file
	region_list = []
	num1 = t_lenvec.shape[0]
	num2 = t_lenvec.ravel().shape[0]
	if num2>3:
		for i in range(0,num1):
			region_list.append(t_lenvec[i])
	else:
		region_list.append(t_lenvec)

	print region_list

	region_id = 0
	type_id = 0
	list1 = []
	for t_position in region_list:
		position1, position2, length = t_position[0], t_position[1], t_position[2]
		size1 = length*1.0/resolution
		print size1

		if size1 > threshold1:
			num1 = int(size1/region_size)
			res = size1-num1*region_size
			if res<threshold2:
				num1 = num1-1
				res = size1-num1*region_size

			for i in range(0,num1+1):
				for j in range(i,num1+1):
					s1, s2 = resolution*region_size*i, resolution*region_size*(i+1)
					s1a, s2a = resolution*region_size*j, resolution*region_size*(j+1)

					if i>=num1:
						s2 = length
					if j>=num1:
						s2a = length
					if i!=j:
						type_id = 1
					else:
						type_id = 0
					
					len1, len2 = s2-s1, s2a-s1a
					print s1,s2,s1a,s2a,len1,len2,region_id,type_id
					list1.append([s1,s2,s1a,s2a,len1,len2,region_id,type_id])
		else:
			list1.append([position1,position2,position1,position2,length,length,region_id,type_id])

		region_id = region_id + 1

	list1 = np.asarray(list1)

	return list1

def subregion1(filename, chrom_id, resolution, region_points, type_id):

	print filename
	t_lenvec = np.loadtxt(filename, dtype='int', delimiter='\t')	# load *.txt file
	print "t_lenvec", t_lenvec
	print "region_points subregion1", region_points

	region_list = []
	num1 = t_lenvec.shape[0]
	num2 = t_lenvec.ravel().shape[0]
	if num2>len(t_lenvec):
		for i in range(0,num1):
			region_list.append(np.hstack((t_lenvec[i,0:3],i)))
		vec1 = np.asarray(region_list)
	else:
		region_list.append(np.hstack((t_lenvec[0:3],0)))
		vec1 = np.asarray(region_list)
		vec1 = vec1.reshape((1,-1))

	print "region_list",region_list
	print "vec1",vec1
	
	region_id = 0
	type_id1 = 0
	threshold = resolution*2

	# vec1 = np.asarray(region_list)
	num1 = len(region_points)

	# if type_id==1:
	# 	for k in range(0,num1):
	# 		vec1 = np.asarray(region_list)
	# 		point1, point2 = region_points[k][0], region_points[k][1]
	# 		temp1 = (vec1[:,0]<point1-threshold)&(vec1[:,1]>point2+threshold)
	# 		b = np.where(temp1==True)[0]

	# 		if len(b)>0:
	# 			id1 = b[0]
	# 			region_id = vec1[id1,3]
	# 			start1, stop1 = vec1[id1,0], point1
	# 			start2, stop2 = point2, vec1[id1,1]
	# 			length1, length2 = stop1 - start1, stop2 - start2
	# 			region_list[id1] = [start2,stop2,length2,region_id]
	# 			region_list.insert(id1,[start1,stop1,length1,region_id])

	for k in range(0,num1):
		vec1 = np.asarray(region_list)
		point1, point2 = region_points[k][0], region_points[k][1]
		temp1 = (vec1[:,0]<point1-threshold)&(vec1[:,1]>point2+threshold)
		b = np.where(temp1==True)[0]

		print "region points", point1, point2, b
		if len(b)>0:
			id1 = b[0]
			region_id = vec1[id1,3]
			start1, stop1 = vec1[id1,0], point1
			start2, stop2 = point2, vec1[id1,1]
			length1, length2 = stop1 - start1, stop2 - start2
			region_list[id1] = [start2,stop2,length2,region_id]
			print "region_list1", region_list
			region_list.insert(id1,[start1,stop1,length1,region_id])
			print "region_list2", region_list			

	region_list1 = np.asarray(region_list)
	region_idvec1 = region_list1[:,-1]
	region_idvec = np.sort(np.unique(region_idvec1))

	list1 = []
	region_id1 = 0

	for region_id in region_idvec:
		b = np.where(region_idvec1==region_id)[0]
		print "region_id", region_id, b
		if len(b)==1:
			t_position = region_list[b[0]]
			position1, position2, length = t_position[0], t_position[1], t_position[2]
			list1.append([position1,position2,position1,position2,length,length,region_id,region_id1,chrom_id])
			region_id1 = region_id1+1
		else:
			n1 = len(b)
			for i in range(0,n1):
				for j in range(i,n1):
					print "i,j", i,j
					t_position1, t_position2 = region_list[b[i]], region_list[b[j]]
					position1, position2, length = t_position1[0], t_position1[1], t_position1[2]
					position1a, position2a, length_1 = t_position2[0], t_position2[1], t_position2[2]
					list1.append([position1,position2,position1a,position2a,length,length_1,region_id,region_id1,chrom_id])
					region_id1 = region_id1+1

	print region_list
	print list1

	# list1 = np.asarray(list1)

	return region_list, list1

# write feature vectors to image
def write_matrix_image_v1(value, pos, output_filename1):

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

	mtx1 = np.zeros((window_size,window_size,dim_feature))
	for i in range(0,n_samples):
		id1, id2 = pos[i,0]-start_region, pos[i,1]-start_region		
		mtx1[id1,id2] = value[i,:]
		if id1>id2:
			print "%d error!"%(i)

		mtx1[id2,id1] = value[i,:]
		if i%100000==0:
			print i, id1, id2, value[i,:]

	# if output_filename1!="":
	# 	np.save(output_filename1,mtx1)

	return mtx1, start_region

# write feature vectors to image with mask matrix used
def write_matrix_image_v1_mask(value, pos, output_filename1):

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

	for i in range(0,dim_feature):
		t1 = value[:,i]
		threshold = np.quantile(t1[t1>0],0.05)
		t1[t1<threshold] = 0
		value[:,i] = t1 

	mtx1 = np.zeros((window_size,window_size,dim_feature))
	for i in range(0,n_samples):
		id1, id2 = pos[i,0]-start_region, pos[i,1]-start_region		
		mtx1[id1,id2] = value[i,:]
		if id1>id2:
			print "%d error!"%(i)

		mtx1[id2,id1] = value[i,:]
		if i%100000==0:
			print i, id1, id2, value[i,:]

	temp1 = np.sum(mtx1,2)

	# filename1 = 'chrom_quantile.txt'
	# #np.savetxt(filename1, m_vec_list, fmt='%.4f', delimiter='\t')

	# vec1 = pd.read_table(filename1,header=None)
	# m_values = vec1[6]
	# x_max = np.median(m_values)
	# print x_max

	threshold1 = 0
	value_index1 = np.where(temp1.ravel()>threshold1)[0]	# the positions with values
	temp1[temp1<=threshold1] = 0

	# if output_filename1!="":
	# 	np.save(output_filename1,mtx1)
	mask = np.ones((window_size,window_size))
	for i in range(1,window_size-1):
		for j in range(i+1,window_size-1):
			if np.sum(temp1[i-1:i+1,j-1:j+1])<=0:
				mask[j,i] = 0
				mask[i,j] = 0

	value_index2 = np.where(mask.ravel()>0)[0]	# the positions with values

	ratio1 = len(value_index1)*1.0/(window_size*window_size)
	ratio2 = len(value_index2)*1.0/(window_size*window_size)

	print "write_matrix_image_v1_mask %.2f %.2f"%(ratio1,ratio2)

	return mtx1, start_region, value_index1, value_index2

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
	# pos_idx_1 = pos_idx[serial,:]

	if output_filename1!="":
		# np.save(output_filename1,data_1)
		np.savetxt(output_filename1, data_1, delimiter='\t', fmt='%.2f')

	return data_1, pos_idx, serial

# write feature vectors to image
def write_matrix_image_v1a(value, pos, output_filename1):

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

	window_size1 = stop_region1-start_region1+1
	window_size2 = stop_region2-start_region2+1
	n_samples, dim_feature = value.shape[0], value.shape[1]
	print "n_samples: %d dim_feature: %d window_size1: %d window_size2: %d start_region1: %d stop_region1: %d start_region2: %d stop_region2: %d"%(n_samples,
															dim_feature, window_size1, window_size2, start_region1, stop_region1, start_region2, stop_region2)

	# mtx1 = np.zeros(n1,n2,dim_feature)
	mtx1 = np.zeros((window_size1,window_size2,dim_feature))
	for i in range(0,n_samples):
		id1, id2 = pos[i,0]-start_region1, pos[i,1]-start_region2		
		mtx1[id1,id2] = value[i,:]
		#if id1>id2:
		#	print "%d error!"%(i)

		#mtx1[id2,id1] = value[i,:]
		if i%100000==0:
			print i, id1, id2, value[i,:]

	# if output_filename1!="":
	# 	np.save(output_filename1,mtx1)

	return mtx1, start_region1, start_region2

# write image to feature vectors
def write_matrix_array_v1a(mtx1, start_region1, start_region2, output_filename1, type_id=0):

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
	# pos_idx = np.asarray(pos_idx)+start_region
	pos_idx = np.asarray(pos_idx)
	pos_idx[:,0] += start_region1
	pos_idx[:,1] += start_region2
	data1 = mtx1.reshape((n1*n2,dim1))
	data_1 = data1[serial,:]

	return data_1, pos_idx, serial

# write feature vectors to image
def write_matrix_image(filename, output_filename1, output_filename2):

	data1 = pd.read_table(filename)
	num1 = data1.shape[0]	# the number of entries
	colnames = list(data1)	# chrom, x1, x2, serial, values

	start_idx = 0
	if 3 in colnames:
		start_idx = 1
	print start_idx 

	pos = np.asarray(data1.loc[:,colnames[start_idx:start_idx+2]])
	value = np.asarray(data1.loc[:,colnames[start_idx+3:]])

	print value.shape

	# n1, n2 = shape1[0], shape1[1]
	n1, n2 = int(np.max(pos[:,0])), int(np.max(pos[:,1]))

	print value.shape

	n_1 = np.max((n1,n2))+1
	print n_1
	dim_feature = len(colnames)-(start_idx+3)

	m1_vec = np.zeros((dim_feature,8))
	value1 = value.copy()
	for i in range(0,dim_feature):
		temp1 = value[:,i]
		b = np.where(temp1<=0)[0]
		m1, m1a, m1a1, std1 = np.mean(temp1), np.median(temp1), np.max(temp1), np.std(temp1)
		temp1[b] = 1e-10	# default small value
		temp2 = np.log2(temp1+1)
		
		m2, m2a, m2a1, std2 = np.mean(temp2), np.median(temp2), np.max(temp2), np.std(temp2)
		m1_vec[i] = [m1,m1a,m1a1,std1,m2,m2a,m2a1,std2]
		value1[:,i] = temp2
	
	print m1_vec
	np.savetxt(output_filename1,m1_vec,fmt='%.2f')
	# value = sklearn.preprocessing.StandardScaler().fit_transform(value1)

	# mtx1 = np.zeros(n1,n2,dim_feature)
	mtx1 = np.zeros((n_1,n_1,dim_feature))
	for i in range(0,num1):
		if i%1000000==0:
			print i
		mtx1[pos[i,0],pos[i,1]] = value[i,:]
		mtx1[pos[i,1],pos[i,0]] = value[i,:]

	np.save(output_filename2,mtx1)

	return mtx1

def compute_serial(x,y,N):

	serial = int((N+1)*x -x*(x+1)/2+y)

	return serial

def quantile_contact_vec(chrom_vec,resolution,ref_filename,filename_list,species):

	m_vec_list = []
	for chrom in chrom_vec:
		print chrom
		m_vec = quantile_contact(chrom,resolution,ref_filename,filename_list, species)
		m_vec_list.extend(m_vec)

	m_vec_list = np.asarray(m_vec_list)

	return m_vec_list

def quantile_contact(chrom,resolution,ref_filename,filename_list, species):

	type_id = 0
	output_filename1 = ""
	# resolution = 50000
	vec1 = multi_contact_matrix3A_single(chrom, resolution, ref_filename, filename_list, species, output_filename1, type_id)
	
	keys_vec = list(vec1.keys())
	print(keys_vec)
	vec2 = [0.05,0.25,0.50,0.75,0.95]
	species_num = len(keys_vec)
	m_vec = np.zeros((species_num,10))
	eps = 1e-16
	species = ['gorGor4','panTro5','panPan2','hg38']
	for i in range(0,species_num):
	# for species_id in keys_vec:
		#species_id = keys_vec[i]
		species_id = species[i]
		temp1 = vec1[species_id]
		values = temp1[2] 
		print(values)
		b1 = np.where(values>0)[0]
		b2 = np.where(values>=0)[0]
		m_vec[i,0:5] = np.percentile(values[b2],vec2)
		m_vec[i,5] = np.min(values[b1])
		m_vec[i,6] = np.max(values)
		m_vec[i,7] = np.max(values)/(m_vec[i,4]+eps)
		m_vec[i,8], m_vec[i,9] = len(b1), len(b2)

	print(m_vec)
	return m_vec

def multi_contact_matrix3A(chrom, resolution, ref_chromsize, filename_list, species, output_filename, type_id):

	filename1 = ref_chromsize
	data1 = pd.read_table(filename1,header=None,sep='\t')
	chrom1, size1 = data1[0], data1[1]
	key1 = "chr%s"%(chrom)
	b1 = np.where(chrom1==key1)[0]
	if len(b1)>0:
		chrom_size = size1[b1[0]]
		N = math.ceil(chrom_size/resolution)
		print N
	else:
		print "chrom size error!"
		return -1

	species_num = len(species)	
	value_dict = dict()
	value_dict1 = dict()
	serial1 = []
	t_serial2 = []

	for i in range(0,species_num):

		input_path = filename_list[i]
		species_id = species[i]

		filename1 = "%s/chr%s.%dK.txt"%(input_path,chrom,int(resolution/1000))
		# filename1 = "%s/chr%s.observed.kr.%dK.txt"%(input_path,chrom,int(resolution/1000))
		if os.path.exists(filename1)==False:
			print "File %s does not exist. Please check."%(filename1)
			return False

		data2 = pd.read_table(filename1,header=None)
		num1 = data2.shape[0]
		x1, x2, value = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2])
		x1, x2 = x1/resolution, x2/resolution

		t_vec_serial = np.int64(N*x1 + x2)
		
		b1 = np.where(np.isnan(value)==True)[0]
		value[b1] = -1
		
		num2 = len(b1)
		print num1, num2

		value_dict[species_id] = t_vec_serial	# serial
		value_dict1[species_id] = [x1,x2,value]	# value

		serial1 = np.union1d(serial1,t_vec_serial)
		t_serial2.append(t_vec_serial)

	serial2 = t_serial2[0]
	print(len(serial2))
	for i in range(1,species_num):
		serial2 = np.intersect1d(serial2,t_serial2[i])

	n1, n2 = len(serial1), len(serial2)
	print "union, intersection", n1, n2

	colnames = [0,1,2]
	colnames.extend(species)
	data_2 = output_multi_contactMtx(serial1, colnames, species, value_dict, value_dict1, resolution, output_filename)
	
	return data_2

def multi_contact_matrix3A_single(chrom, resolution, ref_chromsize, filename_list, species, output_filename, type_id):

	filename1 = ref_chromsize
	data1 = pd.read_table(filename1,header=None,sep='\t')
	chrom1, size1 = data1[0], data1[1]
	key1 = "chr%s"%(chrom)
	b1 = np.where(chrom1==key1)[0]
	if len(b1)>0:
		chrom_size = size1[b1[0]]
		N = math.ceil(chrom_size/resolution)
		print N
	else:
		print "chrom size error!"
		return -1

	species_num = len(species)
	
	value_dict = dict()
	value_dict1 = dict()
	serial1 = []
	t_serial2 = []

	cnt = 3

	species_num = len(species)
	
	value_dict = dict()
	value_dict1 = dict()
	serial1 = []
	t_serial2 = []

	for i in range(0,species_num):

		input_path = filename_list[i]
		species_id = species[i]
		print(species_id)

		# filename1 = "%s/chr%s.%dK.txt"%(input_path,chrom,int(resolution/1000))
		filename1 = "%s/chr%s.observed.kr.%dK.txt"%(input_path,chrom,int(resolution/1000))
		if os.path.exists(filename1)==False:
			print "File %s does not exist. Please check."%(filename1)
			return False

		data2 = pd.read_table(filename1,header=None)
		num1 = data2.shape[0]
		x1, x2, value = np.asarray(data2[0]), np.asarray(data2[1]), np.asarray(data2[2])
		x1, x2 = x1/resolution, x2/resolution

		t_vec_serial = np.int64((N+1)*x1 + x2)

		b1 = np.where(np.isnan(value)==True)[0]
		value[b1] = -1
		
		num2 = len(b1)
		print num1, num2

		value_dict[species_id] = t_vec_serial	# serial
		value_dict1[species_id] = [x1,x2,value]	# value

	return value_dict1

def output_multi_contactMtx(serial1, colnames, species, value_dict, value_dict1, resolution, output_filename1):

	n1 = len(serial1)
	print n1

	data_1 = pd.DataFrame(columns=colnames)

	# data_1 = data1.loc[serial1,colnames]
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

	# data_1.to_csv(output_filename1,header=False,index=False,sep='\t')
	if output_filename1!="":
		data_1.to_csv(output_filename1,index=False,sep='\t')

	return data_1

def chrom_contactMtx(input_filename,chrom):

	str_vec = input_filename.split('.')
	temp1 = str_vec[1].find('Kb')
	resolution = int(str_vec[1][0:temp1])*1000
	temp2 = str_vec[1].find('chr')
	if temp2<0:
		chrom = 'chr%s'%(chrom)
	
	data1 = pd.read_table(input_filename,header=None)
	x1, x2, value1 = data1[0], data1[1], np.asarray(data1[2])
	x1, x2 = x1/resolution, x2/resolution
	
	fields = ['chrom','x1','x2','value']
	data_1 = pd.DataFrame(columns=fields)
	data_1['chrom'] = [chrom]*len(x1)

	b1 = np.where(np.isnan(value1)==True)[0]
	value1[b1] = -1

	data_1['x1'], data_1['x2'],  data_1['value'] = np.int64(x1), np.int64(x2), value1

	temp2 = input_filename.find('.txt')
	output_filename = "%s.bed"%(input_filename[0:temp2])
	data_1.to_csv(output_filename,header=False,index=False,sep='\t')

	return True

def overlap_openChromatin(loc1, loc2):

	chrom1, start1, stop1 = loc1['chr'], loc1['start'], loc1['stop']		# feature region
	# chrom2, start2, stop2 = data2['chr'], data2['start'], data2['stop']		# open chromatin
	# chrom2, start2, stop2 = loc2[1], loc2[3], loc2[4]	# open chromatin
	chrom2, start2, stop2 = loc2[0], loc2[1], loc2[2]	# open chromatin
	
	chrom_vec = list(set(list(chrom1)))
	print chrom_vec
	chrom_num = len(chrom_vec)  # number of chromosomes
	chrom_dict = dict()
	
	for item in chrom_vec:
		idx = np.where(chrom1==item)[0]
		print item, idx
		chrom_dict[item] = idx  # record indices of chromosome

	num1 = len(chrom1)  # number of feature regions
	num2 = len(chrom2)	# number of open chromatin regions
	print chrom_dict
	sel_idx = []
	for j in range(0,num2):
		t_chrom = chrom2[j]
		if t_chrom in chrom_dict:
			b1 = chrom_dict[t_chrom] # index of chromosome
			b2 = (start1[b1]<stop2[j])&(stop1[b1]>start2[j])
			b = b1[np.where(b2==True)[0]]
			if b.shape[0]>0:
				# print start2[j], stop2[j]
				# print np.array((start1[b], stop1[b])).T
				sel_idx.extend(list(b))
	
	sel_idx=list(set(sel_idx))
	return sel_idx

def overlap_openChromatinRegion1(base_id, chrom1, start1, stop1, filename2):

	data2 = pd.read_table(filename2, header=None)	# open chromatin
	chrom2, start2, stop2 = data2[0], np.array(data2[1]), np.array(data2[2])
	pos2 = np.array((start2,stop2)).T

	chrom_vec = list(set(list(chrom2)))
	print chrom_vec
	chrom_num = len(chrom_vec)  # number of chromosomes
	chrom_dict = dict()
	
	for item in chrom_vec:
		idx = np.where(chrom2==item)[0]
		print item, idx
		chrom_dict[item] = idx  # record indices of chromosome

	num1 = chrom1.shape[0]  # number of chromosomes of feature regions
	num2 = chrom2.shape[0]	# number of chromosomes of open chromatin regions
		
	print chrom_dict
	len_vec = np.zeros(num1)
	for j in range(0,num1):
		if j%5000==0:
			print j
		#if j>=1000:
		#	break
		t_chrom = chrom1[j]
		if t_chrom in chrom_dict:
			s1, s2 = start1[j], stop1[j]
			if s1<0 or s2<0:
				continue

			b1 = chrom_dict[t_chrom] # index of open chromatin regions in one chromosome
			b2 = (start2[b1]<s2)&(stop2[b1]>s1)
			b = b1[np.where(b2==True)[0]]
			if b.shape[0]>0:
				tmp = pos2[b]
				tmp[tmp[:,0]<s1,0] = s1
				tmp[tmp[:,1]>s2,1] = s2
				len_vec[j] = np.sum(tmp[:,1]-tmp[:,0])
	
	return len_vec
	
def serial_region(filename,output_filename,output_filename1):

	data1 = pd.read_table(filename,header=None)
	chrom, start, stop = data1[0], data1[1], data1[2]
	num1 = len(chrom)
	data1['serial'] = range(0,num1)
	# data1.to_csv(output_filename,header=False,index=False,sep='\t')

	chrname_vec = list(set(list(chrom)))
	chrname_dict = dict()
	chr_startIdx = pd.DataFrame(columns=['chrom','start'])
	start_Idx = []

	for chrname1 in chrname_vec:
		b1 = (chrom==chrname1)
		idx = np.where(b1==True)[0]
		print chrname1, len(idx)
		chrname_dict[chrname1] = idx
		start_Idx.append(idx[0])

	chr_startIdx['chrom'], chr_startIdx['start'] = chrname_vec, start_Idx
	chr_startIdx.to_csv(output_filename,index=False,header=False,sep='\t')

	return chr_startIdx

def compare_serial(data1,filename2):

	# data1 = pd.read_table(filename1,header=None)
	chrom1, start1, stop1, serial1 = data1[0], data1[1], data1[2], data1[4]

	data2 = pd.read_table(filename2,header=None)
	chrom2, start2, stop2, serial2, serial2a = data2[0], data2[1], data2[2], data2[4], data2[5]
	num1 = len(serial2)
	cnt = 0
	for i in range(0,num1):
		if i%500000==0:
			print i
		id1 = serial2[i]
		t_chrom, t_start, t_stop, t_serial = chrom1[id1], start1[id1], stop1[id1], serial1[id1]
		if (id1!=t_serial) or (t_chrom!=chrom2[i]) or (t_start!=start2[i]):
			print "error!"
			cnt = cnt + 1

		if serial2[i]!=serial2a[i]:
			id1 = serial2a[i]
			t_chrom, t_start, t_stop, t_serial = chrom1[id1], start1[id1], stop1[id1], serial1[id1]
			if (id1!=t_serial) or (t_chrom!=chrom2[i]) or (t_stop!=stop2[i]):
				print "error!"
				cnt = cnt + 1

	print cnt
	return cnt
