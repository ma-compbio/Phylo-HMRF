# Phylogenetic Hidden Markov Random Field Model

import numpy as np
from scipy.misc import logsumexp
from sklearn import cluster
from sklearn import mixture
from sklearn.mixture import (
	sample_gaussian,
	log_multivariate_normal_density,
	distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state

from base import _BaseGraph

import pandas as pd
import numpy as np
import os
import sys
import math
import random
import scipy
import scipy.io

from numpy.linalg import inv, det, norm
from numpy import linalg

from scipy.optimize import minimize

import pickle

import pygco

import sklearn.preprocessing
import multiprocessing as mp

import utility1

from optparse import OptionParser

import os.path

import warnings

import time


__all__ = ["PhyloHMRF"]

COVARIANCE_TYPES = frozenset(("linear","spherical", "diag", "full", "tied"))

small_eps = 1e-16

class phyloHMRF(_BaseGraph):
	
	def __init__(self, n_samples, n_features, n_dim1, n_dim2, edge_list, branch_list, cons_param, beta, beta1, 
				 initial_mode, initial_weight, initial_weight1, initial_magnitude, observation,
				 observation_mtx, edge_list_1, len_vec, type_id = 0, 
				 max_iter = 10,
				 n_components=1, run_id=0, estimate_type=0, covariance_type='full',
				 min_covar=1e-3,
				 startprob_prior=1.0, transmat_prior=1.0, 
				 means_prior=0, means_weight=0,
				 covars_prior=1e-2, covars_weight=1,
				 algorithm="viterbi", random_state=None,
				 n_iter=10, tol=1e-2, verbose=False,
				 params="stmc", init_params="stmc", 
				 learning_rate = 0.001):
		_BaseGraph.__init__(self, n_components=n_components, run_id=run_id, estimate_type=estimate_type, 
						  startprob_prior=startprob_prior,
						  transmat_prior=transmat_prior, algorithm=algorithm,
						  random_state=random_state, n_iter=n_iter,
						  tol=tol, params=params, verbose=verbose,
						  init_params=init_params)

		self.covariance_type = covariance_type
		self.min_covar = min_covar
		self.means_prior = means_prior
		self.means_weight = means_weight
		self.covars_prior = covars_prior
		self.covars_weight = covars_weight
		self.covariance_type = covariance_type
		self.min_covar = min_covar
		self.random_state = random_state
		self.lik = 0
		self.max_iter = max_iter
		self.beta = beta	# regularization coefficient for edge potential
		self.beta1 = beta1	# regularization coefficient for edge potential
		self.type_id = type_id

		self.observation = observation
		self.observation_mtx = observation_mtx
		print "data loaded", self.observation.shape
		print "estimate type %d"%(self.estimate_type)
		
		# self.n_samples, self.n_features = observation.shape
		self.n_samples = n_samples
		self.n_features = n_features
		self.n_components = n_components
		self.learning_rate = learning_rate
		self.n_dim1 = n_dim1
		self.n_dim2 = n_dim2
		#self.img = observation.copy().reshape((n_dim1,n_dim2,n_features))
		#print self.img.shape

		# self.edge_list_1 = edge_list_1	# edge list of the graph
		self.edge_list_vec = edge_list_1	# edge list of the graph
		self.len_vec = len_vec

		self.edge_potential = self._pairwise_potential()
		
		# self.edge_weightList, self.edge_idList_undirected, self.edge_weightList_undirected = self._edge_weight_undirected(observation)

		self.edge_weightList_vec, self.edge_idList_undirected_vec, self.edge_weightList_undirected_vec, \
		self.neighbor_vec2, self.neighbor_edgeIdx_vec, self.edge_index_vec = self._edge_weight_undirected_vec(observation, len_vec, edge_list_1)

		# self.cost_H, self.cost_V = self._edge_weight_grid(self.edge_idList_undirected, self.edge_weightList_undirected)

		self.tree_mtx, self.node_num = self._initilize_tree_mtx(edge_list)
		self.branch_params = branch_list
		# self.branch_dim = len(branch_list)
		self.branch_dim = self.node_num - 1   # number of branches
		
		self.n_params = self.node_num + self.branch_dim*2 + 1  # optimal values (n1), selection strength and variance (n2*2), variance of root node
		
		self.params_vec1 = np.random.rand(n_components, self.n_params) # this needs to be updated

		self.init_ou_params = self.params_vec1.copy()

		print "branch dim", self.branch_dim
		print "number of parameters", self.n_params
		self.branch_vec = [None]*self.node_num  # all the leaf nodes that can be reached from a node

		self.base_struct = [None]*self.node_num
		print "compute base struct"
		self.leaf_list = self._compute_base_struct()
		print self.leaf_list
		self.index_list = self._compute_covariance_index()
		print self.index_list
		self.base_vec = self._compute_base_mtx()
		self.leaf_time = branch_list[0]+branch_list[1]  # this needs to be updated
		self.leaf_vec = self._search_leaf()  # search for the leaves of the tree
		self.path_vec = self._search_ancestor()

		mtx = np.zeros((n_features,n_features))
		print "initilization, branch parameters", self.branch_params
		print "initilization, branch dim:", self.branch_dim
		for i in range(0,self.branch_dim):
			mtx += self.branch_params[i]*self.base_vec[i+1]

		print mtx
		self.cv_mtx = mtx
		print self.leaf_time
		# print self.params_vec1

		#posteriors = np.random.rand(self.n_samples,n_components)
		posteriors = np.ones((self.n_samples,n_components))
		den1 = np.sum(posteriors,axis=1)
		
		# self.posteriors = posteriors/(np.reshape(den1,(self.n_samples,1))*np.ones((1,n_components)))
		self.posteriors = np.ones((self.n_samples,n_components))    # for testing
		self.mean = np.random.rand(n_components, self.n_features)   # for testing
		
		self.stats = dict()
		self.counter = 0

		self.A1, self.A2, self.pair_list, self.parent_list = self._matrix1()

		self.lambda_0 = cons_param  # ridge regression coefficient
		self.initial_mode, self.initial_w1, self.initial_w1a, self.initial_w2 = initial_mode, initial_weight, initial_weight1, initial_magnitude

		print "initial weights", self.initial_w1, self.initial_w1a, self.initial_w2

		print "lambda_0", cons_param, n_samples, self.lambda_0*1.0/np.sqrt(n_samples)

	def _get_covars(self):
		"""Return covars as a full matrix."""
		if self.covariance_type == 'full':
			return self._covars_
		elif self.covariance_type == 'diag':
			return np.array([np.diag(cov) for cov in self._covars_])
		elif self.covariance_type == 'tied':
			return np.array([self._covars_] * self.n_components)
		elif self.covariance_type == 'spherical':
			return np.array(
				[np.eye(self.n_features) * cov for cov in self._covars_])
		elif self.covariance_type == 'linear':
			return self._covars_

	def _set_covars(self, covars):
		self._covars_ = np.asarray(covars).copy()

	covars_ = property(_get_covars, _set_covars)

	def _check(self):
		super(phyloHMRF, self)._check()

		self.means_ = np.asarray(self.means_)
		self.n_features = self.means_.shape[1]

		if self.covariance_type not in COVARIANCE_TYPES:
			raise ValueError('covariance_type must be one of {0}'
							 .format(COVARIANCE_TYPES))

		_validate_covars(self._covars_, self.covariance_type,
						 self.n_components)

	#def _validate_covars_linear(self._covars_, self.covariance_type,
	#                     self.n_components):

	# initialize the parameters of the multiple OU models
	def _init_ou_param(self, X, init_label, mean_values):
		n_components = self.n_components 
		# init_label = self.init_label_.copy()
		init_ou_params = self.params_vec1.copy()

		for i in range(0,n_components):
			b = np.where(init_label==i)[0]
			num1 = b.shape[0]  # number of samples in the initialized cluster
			if num1==0:
				print "empty cluster!"
			else:
				x1 = X[b]
				print "number in the cluster", x1.shape[0], i
				cur_param, lik = self._ou_optimize_init(x1, mean_values[i])
				init_ou_params[i,:] = cur_param.copy()
				print "ou_optimize_init likelihood ", lik 

		print "initial ou paramters"
		print init_ou_params
		
		return init_ou_params

	def _init(self, X, lengths=None):
		super(phyloHMRF, self)._init(X, lengths=lengths)
		
		# _, n_features = X.shape
		dim = X.shape
		n_features = dim[-1]
		self.n_features = n_features
		
		if hasattr(self, 'n_features') and self.n_features != n_features:
			raise ValueError('Unexpected number of dimensions, got %s but '
							 'expected %s' % (n_features, self.n_features))

		self.n_features = n_features
		if 'm' in self.init_params or not hasattr(self, "means_"):
			kmeans = cluster.KMeans(n_clusters=self.n_components,
									random_state=self.random_state,
									max_iter=300, n_jobs=-5, n_init=10)
			kmeans.fit(X)
			self.means_ = kmeans.cluster_centers_
			self.init_label = kmeans.labels_
			self.labels = self.init_label.copy()
			self.labels_local = self.init_label.copy()
			print "initialize parameters..."
			self.init_ou_params = self._init_ou_param(X, self.init_label.copy(), self.means_.copy())
			self.params_vec1 = self.init_ou_params.copy()

		if 'c' in self.init_params or not hasattr(self, "covars_"):
			# cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
			cv = np.cov(X.T) + self.min_covar * np.eye(n_features)
			if not cv.shape:
				cv.shape = (1, 1)
			self._covars_ = distribute_covar_matrix_to_match_covariance_type(
				cv, self.covariance_type, self.n_components).copy()

		print "return from initializing parameters..."

	def _compute_log_likelihood(self, X):
		return log_multivariate_normal_density(
			X, self.means_, self._covars_, self.covariance_type)

	def _compute_posteriors_graph1(self, X, label):
		prob = np.exp(self.logprob)	# self.logprob should have been updated with the current model parameters
		pairwise_prob = self._pairwise_compare(label)
		weighted_prob = prob*pairwise_prob
		sum1 = np.sum(weighted_prob,axis=1).reshape((-1,1))
		temp1 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		posteriors = weighted_prob/temp1

		return posteriors

	def _compute_posteriors_graph_v1(self, X, label):
		pairwise_potential = self._pairwise_compare(label)
		weighted_prob = np.exp(self.logprob - pairwise_potential)
		sum1 = np.sum(weighted_prob,axis=1).reshape((-1,1))
		temp1 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		posteriors = weighted_prob/temp1

		self.q_edge = np.sum(posteriors*pairwise_potential)
		self.pairwise_potential = pairwise_potential.copy()

		return posteriors

	def _compute_posteriors_graph_test(self,len_vec, X, region_id, posteriors_test, posteriors_test1):

		s1, s2 = len_vec[region_id][1], len_vec[region_id][2]
		temp1 = np.mean(X[s1:s2],0)
		posteriors_test = posteriors_test + temp1
		posteriors_test1.append(temp1)

		self.queue.put((region_id,s1,s2,temp1))
		print region_id,s1,s2,temp1

		time.sleep(5)

		return (s1,s2,temp1)

	def _compute_posteriors_graph_v1(self, X, label, logprob, region_id):

		self.neighbor_vec = self.neighbor_vec2[region_id]
		self.edge_index = self.edge_index_vec[region_id]
		self.neighbor_edgeIdx = self.neighbor_edgeIdx_vec[region_id]

		self.edge_weightList = self.edge_weightList_vec[region_id]

		print "region %d neighbor_vec %d"%(region_id, len(self.neighbor_vec))

		pairwise_potential = self._pairwise_compare(label)
		# print "pairwise_potential", pairwise_potential.shape
		# weighted_prob = np.exp(self.logprob - pairwise_potential)
		weighted_prob = np.exp(logprob - pairwise_potential)
		sum1 = np.sum(weighted_prob,axis=1).reshape((-1,1))
		temp1 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		posteriors = weighted_prob/temp1

		self.q_edge = np.sum(posteriors*pairwise_potential)
		self.pairwise_potential = pairwise_potential.copy()

		pairwise_prob = np.exp(-pairwise_potential)
		print "pairwise_potential, pairwise_prob",pairwise_potential.shape, pairwise_prob.shape
		sum1 = np.sum(pairwise_prob,axis=1).reshape((-1,1))
		temp2 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		pairwise_prob_normalize = pairwise_prob/temp2
		self.pairwise_prob = pairwise_prob_normalize	# normalized pairwise probability

		pairwise_cost, pairwise_cost_normalize, unary_cost, cost1 = self._compute_cost_v1(X, label, logprob)
		
		return posteriors, pairwise_cost, pairwise_cost_normalize, unary_cost, cost1

	def _predict_posteriors(self, X, len_vec, region_id, m_queue):

		s1, s2 = len_vec[region_id][1], len_vec[region_id][2]
		print region_id,s1,s2
		labels, logprob = self.predict(X[s1:s2],region_id)
		posteriors, t_pairwise_cost1, t_pairwise_cost, t_unary_cost, t_cost1 = self._compute_posteriors_graph(X[s1:s2],labels,logprob,region_id)

		stats = dict()
		stats['post'] = posteriors.sum(axis=0)
		stats['obs'] = np.dot(posteriors.T, X[s1:s2])
		# stats['obs**2'] = np.zeros((self.n_components, self.n_features))
		stats['obs*obs.T'] = np.einsum('ij,ik,il->jkl', posteriors, X[s1:s2], X[s1:s2])

		print "stats post", stats['post']

		# framelogprob = self.logprob
		# self._accumulate_sufficient_statistics(stats, X[s1:s2], logprob, posteriors)
		m_queue.put((region_id, stats, labels, t_pairwise_cost1, t_pairwise_cost, t_unary_cost, t_cost1))

		print "return from region %d"%(region_id)

		return True

	def _predict_posteriors1(self, X, len_vec, region_id, m_queue):

		s1, s2 = len_vec[region_id][1], len_vec[region_id][2]
		print region_id,s1,s2
		labels, logprob = self.predict(X[s1:s2],region_id)
		posteriors = self._compute_posteriors_graph1(X[s1:s2],labels,logprob,region_id)

		# framelogprob = self.logprob
		# self._accumulate_sufficient_statistics(stats, X[s1:s2], logprob, posteriors)
		m_queue.put((region_id, labels, posteriors))

		return True

	def _compute_posteriors_graph(self, X, label, logprob, region_id):

		neighbor_vec = self.neighbor_vec2[region_id]
		edge_index = self.edge_index_vec[region_id]
		neighbor_edgeIdx =self.neighbor_edgeIdx_vec[region_id]

		edge_weightList = self.edge_weightList_vec[region_id]

		print "region %d neighbor_vec %d"%(region_id, len(neighbor_vec))

		pairwise_potential = self._pairwise_compare(label, neighbor_vec, edge_index, edge_weightList)
		# print "pairwise_potential", pairwise_potential.shape
		# weighted_prob = np.exp(self.logprob - pairwise_potential)
		weighted_prob = np.exp(logprob - pairwise_potential)
		sum1 = np.sum(weighted_prob,axis=1).reshape((-1,1))
		temp1 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		posteriors = weighted_prob/temp1

		#self.q_edge = np.sum(posteriors*pairwise_potential)
		#self.pairwise_potential = pairwise_potential.copy()

		pairwise_prob = np.exp(-pairwise_potential)
		print "pairwise_potential, pairwise_prob",pairwise_potential.shape, pairwise_prob.shape
		sum1 = np.sum(pairwise_prob,axis=1).reshape((-1,1))
		temp2 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		pairwise_prob_normalize = pairwise_prob/temp2
		# self.pairwise_prob = pairwise_prob_normalize	# normalized pairwise probability

		pairwise_cost, pairwise_cost_normalize, unary_cost, cost1 = self._compute_cost_v1(X, label, logprob, 
						pairwise_prob_normalize, neighbor_vec, neighbor_edgeIdx, edge_index, edge_weightList)
		
		return posteriors, pairwise_cost, pairwise_cost_normalize, unary_cost, cost1

	def _compute_posteriors_graph1(self, X, label, logprob, region_id):
		#log_prob = log_multivariate_normal_density(
		#	X, self.means_, self._covars_, self.covariance_type)

		# prob = np.exp(self.logprob)	# self.logprob should have been updated with the current model parameters
		# pairwise_prob = self._pairwise_compare(label)

		neighbor_vec = self.neighbor_vec2[region_id]
		edge_index = self.edge_index_vec[region_id]
		neighbor_edgeIdx =self.neighbor_edgeIdx_vec[region_id]

		edge_weightList = self.edge_weightList_vec[region_id]

		print "region %d neighbor_vec %d"%(region_id, len(neighbor_vec))

		pairwise_potential = self._pairwise_compare(label, neighbor_vec, edge_index, edge_weightList)
		# print "pairwise_potential", pairwise_potential.shape
		# weighted_prob = np.exp(self.logprob - pairwise_potential)
		weighted_prob = np.exp(logprob - pairwise_potential)
		sum1 = np.sum(weighted_prob,axis=1).reshape((-1,1))
		temp1 = np.dot(sum1,1.0*np.ones((1,self.n_components)))
		posteriors = weighted_prob/temp1
		
		return posteriors

	def _compute_cost(self, X, label, logprob_1):

		print "compute cost..."
		pairwise_cost = self._pairwise_compare_ensemble(label)
		# posteriors = self._compute_posteriors_graph(X,label)
		# unary_cost = -np.sum(self.logprob)*1.0/self.n_samples	# self.logprob should have been updated with the current model parameters
		unary_cost = 0
		# n_samples, n_components = self.n_samples, self.n_components
		n_samples, n_components = X.shape[0], self.n_components
		mask = np.zeros((n_samples,n_components))

		for i in range(0,n_samples):
			mask[i,label[i]] = 1
		# logprob = self.logprob.copy()*mask
		logprob = logprob_1.copy()*mask  # logprob_1 has been calculated

		# print self.logprob[0:5]
		print logprob[0:5]

		unary_cost = np.sum(logprob,axis=1)
		unary_cost = np.sum(unary_cost)

		unary_cost = -unary_cost*1.0/self.n_samples

		return pairwise_cost, unary_cost

	def _compute_cost_v1(self, X, label, logprob1, pairwise_prob_normalize, neighbor_vec, neighbor_edgeIdx, edge_index, edge_weightList):

		print "compute cost..."
		pairwise_cost = self._pairwise_compare_ensemble(label,neighbor_vec,neighbor_edgeIdx,edge_weightList)
		# posteriors = self._compute_posteriors_graph(X,label)
		# unary_cost = -np.sum(self.logprob)*1.0/self.n_samples	# self.logprob should have been updated with the current model parameters
		unary_cost = 0
		# n_samples, n_components = self.n_samples, self.n_components
		n_samples, n_components = len(X), self.n_components
		mask = np.zeros((n_samples,n_components))

		for i in range(0,n_samples):
			mask[i,label[i]] = 1
		
		# logprob = self.logprob.copy()*mask
		logprob = logprob1.copy()*mask
		pairwise_prob_normalize1 = np.log(pairwise_prob_normalize+small_eps)*mask   # self.pairwise_prob should have been updated with the current model parameters
		# print self.logprob[0:5]
		print logprob[0:5]

		unary_cost = np.sum(logprob,axis=1)
		unary_cost = -np.sum(unary_cost)*1.0/n_samples

		pairwise_cost_normalize1 = -np.sum(pairwise_prob_normalize1)*1.0/n_samples

		cost1 = unary_cost+pairwise_cost_normalize1

		return pairwise_cost, pairwise_cost_normalize1, unary_cost, cost1

	def _pairwise_compare(self, label, neighbor_vec, edge_index, edge_weightList):

		n_samples = len(label)
		print "pairwise compare %d"%(n_samples)

		#n = label.shape[0]	# the number of bins
		cost_vec = []
		for i in range(0,n_samples):
			# cost = self._pairwise_compare2(label1, i, j, n_dim1)
			edge_potential = self._pairwise_compareLocal(label, i, neighbor_vec, edge_index, edge_weightList)
			cost_vec.append(edge_potential)
		
		cost_vec = np.asarray(cost_vec)

		return cost_vec

	def _pairwise_compareLocal(self, label, i, neighbor_vec, edge_index, edge_weightList):
		
		flag = 0

		n_components = self.n_components
		idx = neighbor_vec[i]
		vec1 = np.asarray(label[idx])
		
		edge_potential = np.zeros(n_components)
		for k in idx:
			# t_idx.append(self.edge_index[(k,i)])
			# id2 = self.edge_index[(k,i)]
			id2 = edge_index[(k,i)]
			# t_idx.append(id2)
			state1 = label[k]

			if self.estimate_type==3:
				# edge_potential = edge_potential + self.edge_potential[state1]*self.edge_weightList[id2]
				edge_potential = edge_potential + self.edge_potential[state1]*edge_weightList[id2]
			else:
				edge_potential = edge_potential + self.edge_potential[state1]

		t_edge = edge_potential	# messages from the neighbors of node j other than node i

		return t_edge

	# grid-like structure
	# compute the cost based on estimated labels
	def _pairwise_compare_ensemble_v1(self, label):
		
		n_dim1, n_dim2 = self.n_dim1, self.n_dim2
		label1 = label.reshape(n_dim1,n_dim2)
		#n = label.shape[0]	# the number of bins
		cost_vec = []
		for i in range(0,n_dim1):
			for j in range(0,n_dim2):
				cost = self._pairwise_compare_single(label1, i, j, n_dim1)
				cost_vec.append(cost)
		
		cost_vec = np.asarray(cost_vec)

		return np.sum(cost_vec)*1.0/self.n_samples

	# edge_list used
	# compute the cost based on estimated labels
	def _pairwise_compare_ensemble(self, label, neighbor_vec, neighbor_edgeIdx, edge_weightList):
		
		n_samples = len(label)
		#n = label.shape[0]	# the number of bins
		cost_vec = np.zeros(n_samples)
		for i in range(0,n_samples):
			edge_potential = self._pairwise_compare_single(label,i,neighbor_vec,neighbor_edgeIdx,edge_weightList)
			cost_vec[i] = edge_potential

		return np.sum(cost_vec)*1.0/n_samples

	# edge_list used
	# compute the cost based on estimated labels
	def _pairwise_compare_single(self, label, i, neighbor_vec, neighbor_edgeIdx, edge_weightList):
		
		# print "_pairwise_compare_single"

		n_components1 = self.n_components
		idx = neighbor_vec[i]
		vec1 = np.asarray(label[idx])
		t_label = label[i]

		t_idx = neighbor_edgeIdx[i]

		edge_potential = self.edge_potential[vec1,t_label]

		if self.estimate_type==3:
			# edge_weight = self.edge_weightList[t_idx]
			edge_weight = edge_weightList[t_idx]
			# mtx1 = np.dot(edge_weight.reshape((len(edge_weight),1)),np.ones((1,n_components1)))		
			edge_potential = edge_potential*edge_weight

		# edge_potential = self.edge_potential[vec1,t_label]

		b = np.where(np.isnan(edge_potential))[0]
		if len(b)>0:
			print "_pairwise_compare_single nan %d"%(len(b))

		return sum(edge_potential)

	def predict_1(self, X, params_vec):
		
		self.params_vec1 = params_vec.copy()
		self._ou_param_varied_constraint(params_vec)

		num_region = len(self.len_vec)
		state_est1 = np.zeros(X.shape[0])
		for region_id in range(0,num_region):
			s1, s2 = self.len_vec[region_id][1], self.len_vec[region_id][2]
			print(s1,s2)
			labels, logprob = self.predict(X[s1:s2],region_id)
			state_est1[s1:s2] = labels

		return state_est1

	def predict(self, X, region_id):

		# print "predicting states..."

		n_samples, id1, id2, n_dim1 = self.len_vec[region_id][0], self.len_vec[region_id][1], self.len_vec[region_id][2], self.len_vec[region_id][3]

		edge_idList_undirected = self.edge_idList_undirected_vec[region_id]
		edge_weightList_undirected = self.edge_weightList_undirected_vec[region_id]
		# init_labels = self.labels[id1:id2].copy()
		init_labels = self.labels_local[id1:id2].copy()
		state, logprob = self._estimate_state_graphcuts_gco(X,init_labels,edge_idList_undirected,edge_weightList_undirected)

		state1 = state.reshape((n_dim1,n_dim1))
		state = utility1.symmetric_state(state1)
		state = state.reshape(n_samples)

		self.labels[id1:id2] = state
		# self.logprob[id1:id2] = logprob
		
		return state, logprob

	def _do_func(self, framelogprob):

		print "predicting states..."
		#state = self._estimate_state_graphcuts(X)
		type_id = 0
		state = self._estimate_state_LBP(X,type_id)

	def _potential_estimate(self, X):

		self.logprob = self._compute_log_likelihood(X)
		logprob = self.logprob.copy()

	def _estimate_state_graphcuts(self, X, init_labels1):
		
		print "estimating with graph cuts..."
		self.logprob = self._compute_log_likelihood(X)
		weights = -self.logprob.copy()

		print self.n_dim1, self.n_dim2
		D = weights.reshape((self.n_dim1,self.n_dim2,self.n_components))
		max_cycles1 = 5000
		print D.shape

		V = self.edge_potential

		labels = abswap_grid(D,V,max_cycles=max_cycles1,labels=init_labels1)

		vec1 = []
		for i in range(0,self.n_components):
			b1 = np.where(labels==i)[0]
			vec1.append(len(b1))
		
		print vec1

		return labels.reshape(self.n_samples)
	
	def _estimate_state_graphcuts_gco(self, X, init_labels1, edge_idList_undirected, edge_weightList_undirected):
		
		print "estimating with graph cuts general gco..."

		logprob = self._compute_log_likelihood(X)
		unary_cost1 = -logprob.copy()

		print "unary cost max min median mean %.2f %.2f %.2f %.2f"%(np.max(unary_cost1),np.min(unary_cost1),np.median(unary_cost1),np.mean(unary_cost1))

		# print self.n_dim1, self.n_dim2
		# unary_cost1 = unary_cost1.reshape((self.n_dim1,self.n_dim2,self.n_components))
		max_cycles1 = 5000
		print unary_cost1.shape

		V = self.edge_potential

		labels = pygco.cut_general_graph(edge_idList_undirected, edge_weightList_undirected, unary_cost1, V,
					  n_iter=max_cycles1, algorithm='swap', init_labels=init_labels1,
					  down_weight_factor=None)
		#labels = pygco.cut_general_graph(edges, edge_weights, unary_cost1, V,
		#              n_iter=-1, algorithm='expansion', init_labels=labels1,
		#              down_weight_factor=None)

		vec1 = []
		for i in range(0,self.n_components):
			b1 = np.where(labels==i)[0]
			vec1.append(len(b1))
		
		print vec1

		return labels, logprob

	def _estimate_state_graphcuts_gcoGrid(self, X):
		
		print "estimating with graph cuts gco grid..."
		#img = self.X
		#n1, n2, dim = img.shape[0], img.shape[1], img.shape[2]
		#X1 = img.reshape(n1*n2)
		self.logprob = self._compute_log_likelihood(X)
		unary_cost1 = -self.logprob.copy()

		#x_min, x_max = 0, 10
		#weights = utility1.normalize_feature1(weights,x_min,x_max)

		print "unary cost max min median mean %.2f %.2f %.2f %.2f"%(np.max(unary_cost1),np.min(unary_cost1),np.median(unary_cost1),np.mean(unary_cost1))

		print self.n_dim1, self.n_dim2
		unary_cost1 = unary_cost1.reshape((self.n_dim1,self.n_dim2,self.n_components))
		max_cycles1 = 5000
		print unary_cost1.shape

		#V = self.edge_potential
		pairwise_cost1 = self.edge_potential

		# algorithm: `expansion` or `swap`, default=expansion, Whether to perform alpha-expansion or alpha-beta-swaps.
		labels = pygco.cut_grid_graph_simple(unary_cost1, pairwise_cost1, n_iter=max_cycles1, algorithm='swap')
		# labels = pygco.cut_grid_graph_simple(unary_cost1, pairwise_cost1, n_iter=max_cycles1, algorithm='expansion')
		# labels = pygco.cut_grid_graph(unary_cost1, pairwise_cost1, self.cost_V, self.cost_H, n_iter=-1, algorithm='expansion')

		vec1 = []
		for i in range(0,self.n_components):
			b1 = np.where(labels==i)[0]
			vec1.append(len(b1))
		
		print vec1

		return labels.reshape(self.n_samples)

	def _pairwise_prob(self):

		n_components = self.n_components
		beta = self.beta # regularization coefficient
		edge_prob = np.ones((n_components,n_components))
		for i in range(0,n_components):
			for j in range(i+1,n_components):
				edge_potential = beta
				edge_prob[i,j] = np.exp(-edge_potential)
				edge_prob[j,i] = edge_prob[i,j]
				
			edge_prob[i] = edge_prob[i]*1.0/sum(edge_prob[i])

		# self.edge_prob = edge_prob

		return edge_prob

	def _pairwise_potential(self):

		n_components = self.n_components
		beta = self.beta # regularization coefficient
		edge_potential = np.zeros((n_components,n_components))
		for i in range(0,n_components):
			for j in range(i+1,n_components):
				edge_potential[i,j] = beta
				edge_potential[j,i] = edge_potential[i,j]

		self.edge_potential = edge_potential

		return edge_potential

	# construct connections based on grid-like structure
	def _connected_grid(self):

		n_samples = self.n_samples
		neighbor_vec = [None]*n_samples
		neighbor_edgeIdx = [None]*n_samples
		edge_index = dict()
		id1 = 0
		for i in range(0,n_samples):
			vec1 = self._neighbors(i)
			neighbor_vec[i] = vec1
			vec2 = []
			for j in vec1:
				edge_index[(j,i)] = id1
				vec2.append(id1)
				id1 = id1+1
			neighbor_edgeIdx[i] = vec2

		self.neighbor_vec = neighbor_vec
		self.neighbor_edgeIdx = neighbor_edgeIdx
		self.edge_index = edge_index
		self.n_edges = id1

		return True

	# penalty based on the difference of features of adjacent vertices
	def _edge_weight(self, X):

		n_samples = self.n_samples
		n_edges = len(self.edge_list_1)
		edge_list_1 = self.edge_list_1
		beta1 = self.beta1
		edge_weightList = np.zeros(n_edges)

		X_norm = np.sqrt(np.sum(X*X,axis=1))
		
		for k in range(0,n_edges):
			j, i = edge_list_1[k,0], edge_list_1[k,1]
			x1, x2 = X[j], X[i]
			difference = np.dot(x1-x2,(x1-x2).T)
			temp1 = difference/(X_norm[i]*X_norm[j])
			edge_weightList[k] = np.exp(-beta1*temp1)

		#self.edge_weightList = edge_weightList

		print edge_weightList.shape

		filename = 'edge_weightList.txt'
		np.savetxt(filename, edge_weightList, fmt='%.4f', delimiter='\t')

		return edge_weightList

	def _edge_weight_undirected_vec(self, X, len_vec, edge_list_vec):

		print "edge weight undirected"

		num_region = len(len_vec)
		edge_weightList_vec = [None]*num_region
		edge_idList_undirected_vec = [None]*num_region
		edge_weightList_undirected_vec = [None]*num_region

		neighbor_vec2 = [None]*num_region
		neighbor_edgeIdx_vec = [None]*num_region
		edge_index_vec = [None]*num_region

		for id1 in range(0,num_region):
			
			n_samples,i,j = len_vec[id1][0], len_vec[id1][1], len_vec[id1][2]
			edge_list = edge_list_vec[id1]

			print "%d %d %d %d"%(id1,i,j,len(edge_list))

			edge_weightList, edge_idList_undirected, edge_weightList_undirected = self._edge_weight_undirected(X[i:j], edge_list)

			neighbor_vec, neighbor_edgeIdx, edge_index = self._connected_edge(edge_list,n_samples)

			edge_weightList_vec[id1] = edge_weightList
			edge_idList_undirected_vec[id1] = edge_idList_undirected
			edge_weightList_undirected_vec[id1] = edge_weightList_undirected

			neighbor_vec2[id1] = neighbor_vec
			neighbor_edgeIdx_vec[id1] = neighbor_edgeIdx
			edge_index_vec[id1] = edge_index

		return edge_weightList_vec, edge_idList_undirected_vec, edge_weightList_undirected_vec, neighbor_vec2, neighbor_edgeIdx_vec, edge_index_vec

	def _edge_weight_undirected(self, X, edge_list_1):

		print "edge weight undirected"
		
		# n_samples = self.n_samples
		# n_edges = len(self.edge_list_1)
		# edge_list_1 = self.edge_list_1
		beta1 = self.beta1
		n_samples = len(X)
		n_edges = len(edge_list_1)
		edge_weightList = np.zeros(n_edges)

		if n_edges%2!=0:
			print "number of edges error! %d"%(n_edges)
		
		n_edges1 = int(n_edges/2)

		eps = 1e-16 
		X_norm = np.sqrt(np.sum(X*X,axis=1))
		cnt = 0
		for k in range(0,n_edges):
			j, i = edge_list_1[k,0], edge_list_1[k,1]
			x1, x2 = X[j], X[i]
			difference = np.dot(x1-x2,(x1-x2).T)
			temp1 = difference/(X_norm[i]*X_norm[j]+small_eps)
			
			edge_weightList[k] = np.exp(-beta1*temp1)

		b = np.where(edge_list_1[:,0]<edge_list_1[:,1])[0]
		print "id1<id2:%d"%(len(b))
		edge_idList_undirected = edge_list_1[b]
		edge_weightList_undirected = edge_weightList[b]

		#self.edge_weightList = edge_weightList

		print edge_weightList.shape

		position1 = edge_list_1[0,0]
		filename = 'edge_weightList%d.txt'%(position1)
		np.savetxt(filename, edge_weightList, fmt='%.4f', delimiter='\t')

		filename = 'edge_weightList_undirected%d.txt'%(position1)
		fields = ['start','stop','weight']
		data1 = pd.DataFrame(columns=fields)
		data1['start'], data1['stop'] = edge_idList_undirected[:,0], edge_idList_undirected[:,1]
		data1['weight'] = edge_weightList_undirected
		data1.to_csv(filename,header=False,index=False,sep='\t')

		print "edge weight output to file"

		return edge_weightList, edge_idList_undirected, edge_weightList_undirected

	def _edge_weight_grid(self, edge_idList, edge_weightList):

		temp1, temp2 = np.max(edge_idList[:,0]), np.max(edge_idList[:,1])
		n_dim = np.max((temp1,temp2))+1

		temp3 = (edge_idList[:,1]-edge_idList[:,0])==1
		b1 = np.where(temp3==True)[0]
		b2 = np.where(temp3==False)[0]
		height, width = n_dim, n_dim

		t_costH = np.ones((height*width))
		t_costV = np.ones((height*width))
		print height*(width-1), len(b1), (height-1)*width, len(b2) 

		id1 = edge_idList[b1,0]
		t_costH[id1] = edge_weightList[b1]
		id2 = edge_idList[b2,0]
		t_costV[id2] = edge_weightList[b2]

		t_costH = t_costH.reshape((height,width))
		t_costV = t_costV.reshape((height,width))

		return t_costH[:,0:-1], t_costV[0:-1,:]
	
	# construct connections from edge list
	def _connected_edge(self,edge_list_1,n_samples):

		# n_samples = self.n_samples
		neighbor_vec = [None]*n_samples
		neighbor_edgeIdx = [None]*n_samples

		edge_index = dict()
		# n_edges = len(self.edge_list_1)
		n_edges = len(edge_list_1)

		for i in range(0,n_samples):
			neighbor_vec[i] = []
			neighbor_edgeIdx[i] = []

		for id1 in range(0,n_edges):
			#t_edge = self.edge_list_1[id1]
			t_edge = edge_list_1[id1]
			# i,j = t_edge[0], t_edge[1]	# j->i
			j,i = t_edge[0], t_edge[1]	# j->i 
			edge_index[(j,i)] = id1
			neighbor_vec[i].append(j)
			neighbor_edgeIdx[i].append(id1)

		return neighbor_vec, neighbor_edgeIdx, edge_index

	def _generate_sample_from_state(self, state, random_state=None):
		if self.covariance_type == 'tied':
			cv = self._covars_
		else:
			cv = self._covars_[state]
		return sample_gaussian(self.means_[state], cv, self.covariance_type,
							   random_state=random_state)

	def _initialize_sufficient_statistics(self):
		stats = super(phyloHMRF, self)._initialize_sufficient_statistics()
		stats['post'] = np.zeros(self.n_components)
		stats['obs'] = np.zeros((self.n_components, self.n_features))
		stats['obs**2'] = np.zeros((self.n_components, self.n_features))
		stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
										   self.n_features))
		return stats

	def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
										  posteriors):
		super(phyloHMRF, self)._accumulate_sufficient_statistics(
			stats, obs, framelogprob, posteriors)

		if 'm' in self.params or 'c' in self.params:
			stats['post'] += posteriors.sum(axis=0)
			stats['obs'] += np.dot(posteriors.T, obs)

		if 'c' in self.params:
			stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
			stats['obs*obs.T'] += np.einsum(
					'ij,ik,il->jkl', posteriors, obs, obs)

	def params_Initialization(self, state_num, branch_dim):
		params = {
			'v_branch': weight_init([state_num, branch_dim])
		}

		print "parameters intialized"
		return params

	# initilize the connected graph of the tree given the edges
	def _initilize_tree_mtx(self, edge_list):
		node_num = np.max(np.max(edge_list))+1  # number of nodes; index starting from 0
		tree_mtx = np.zeros((node_num,node_num))
		for edge in edge_list:
			p1, p2 = np.min(edge), np.max(edge)
			tree_mtx[p1,p2] = 1

		print "tree matrix built"
		print tree_mtx

		return tree_mtx, node_num

	# find all the leaf nodes which can be reached from a given node
	def _sub_tree_leaf(self, index):
		tmp = self.tree_mtx[index] # find the neighbors
		idx = np.where(tmp==1)[0]
		print idx
		print "size of branch vec", len(self.branch_vec)

		node_vec = []
		if idx.shape[0]==0:
			node_vec = [index]  # the leaf node 
			print "leaf", node_vec
		else:
			for j in idx:
				node_vec1 = self._sub_tree_leaf(j)
				node_vec = node_vec + node_vec1
				print "interior", node_vec

		self.branch_vec[index] = node_vec  # all the leaf nodes that can be reached from this node
		
		print "branch_dim", index, node_vec

		return node_vec

	# find all the pairs of leaf nodes which has a given node as the nearest common ancestor
	def _compute_base_struct(self):  
		node_num = self.node_num
		node_vec = self._sub_tree_leaf(0)  # start from the root node
		cnt = 0
		leaf_list = dict()

		for i in range(0,node_num):
			list1 = self.branch_vec[i]  # all the leaf nodes that can be reached from this node
			num1 = len(list1)
			if num1 == 1:
				leaf_list[i] = cnt
				cnt +=1
			self.base_struct[i] = []
			for j in range(0,num1):
				for k in range(j,num1):
					self.base_struct[i].append(np.array((list1[j],list1[k])))

		print "index built"
		if node_num>2:
			print self.branch_vec[1], self.branch_vec[2]
		return leaf_list

	# find the pair of nodes that share a node as common ancestor
	def _compute_covariance_index(self):
		index = []
		num1 = self.node_num    # the number of nodes
		for k in range(0,num1): # starting from index 1
			t_index = []
			leaf_vec = self.base_struct[k]   # the leaf nodes that share this ancestor
			num2 = len(leaf_vec)
			for i in range(0,num2):
				id1, id2 = self.leaf_list[leaf_vec[i][0]], self.leaf_list[leaf_vec[i][1]]
				t_index.append([id1,id2])
				if id1!=id2:
					t_index.append([id2,id1])
			index.append(t_index)

		return index

	# compute base matrix
	def _compute_base_mtx(self):
		base_vec = dict()
		index, n_features = self.index_list, self.n_features
		num1 = len(index)
		print "index size", index
		base_vec[0] = np.ones((n_features,n_features)) # base matrix for the root node
		for i in range(1,num1):
			indices = index[i]
			cv = np.zeros((n_features,n_features))
			num2 = len(indices)
			for j in range(0,num2):
				id1 = indices[j]
				cv[id1[0],id1[1]] = 1
				cv[id1[1],id1[0]] = 1  # symmetric matrix
			base_vec[i] = cv

		for i in range(0,num1):
			filename = "base_mtx_%d"%(i)
			np.savetxt(filename, base_vec[i], fmt='%d', delimiter='\t')

		return base_vec

	# compute covariance matrix for a state
	def _compute_covariance_mtx_2(self, params):
		# cv1 = self._covars_     # covariance matrix
		# branch_dim, n_features = self.branch_dim, self.n_features
		mtx = self.cv_mtx
		T = self.leaf_time
		alpha, sigma = params[0], params[1]
		a1 = 2.0*alpha
		# mtx1 = np.exp(-a1*(T-cv))
		# mtx2 = 1-np.exp(-a1*cv
		sigma1 = sigma**2
		cv = sigma**2/a1*np.exp(-a1*(T-mtx))*(1-np.exp(-a1*mtx))

		return cv

	def _compute_log_likelihood_2(self, params, state_id):
		cv, n_samples = self._compute_covariance_mtx_2(params), self.n_samples
		inv_cv = inv(cv)
		weights_sum = self.stats['post'][state_id]
		#Sn = self.Sn_w[state_id]/n_samples
		#likelihood = weights_sum*np.log(det(cv))/n_samples+np.matrix.trace(np.matmul(Sn,inv_cv))

		T = cv[0,0]
		a1 = 2.0*alpha
		# mtx1 = np.exp(-a1*(T-cv))
		# mtx2 = 1-np.exp(-a1*cv
		sigma1 = sigma**2
		V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
		s1 = np.exp(-alpha*T)
		print theta0
		print theta1
		theta = theta0*s1+theta1*(1-s1)
		print theta
		n,d = data1.shape[0], data1.shape[1]

		x1 = data1-theta
		lik = weights_sum*np.log(det(V))/n_samples+np.sum(inv(V)*np.matmul(x1.T,x1))/n
		
		return likelihood

	def _output_stats(self, number):
		filename = "log1/stats_iter_%d"%(number)
		np.savetxt(filename, self.stats['post'], fmt='%.4f', delimiter='\t')

	def _ou_lik(self, params, cv, state_id):
		
		alpha, sigma, theta0, theta1 = params[0], params[1], params[2], params[3:]
		T = self.leaf_time
		a1 = 2.0*alpha
		
		# sigma1 = sigma**2
		# V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
		V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
		s1 = np.exp(-alpha*T)
		# print theta0, theta1, theta
		theta = theta0*s1+theta1*(1-s1)
		c = state_id
		obsmean = np.outer(self.stats['obs'][c], theta)

		Sn_w = (self.stats['obs*obs.T'][c]
				- obsmean - obsmean.T
				+ np.outer(theta, theta)*self.stats['post'][c])

		n_samples = self.n_samples
		# weights_sum = stats['post'][c]
		lik = self.stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

		return lik

	# search all the ancestors of a leaf node   
	def _search_ancestor(self):
		path_vec = []        
		tree_mtx = self.tree_mtx
		# n = tree_mtx.shape[0]
		n = self.leaf_vec.shape[0]
		for i in range(0,n):
			leaf_idx = self.leaf_vec[i]
			b = np.where(tree_mtx[:,leaf_idx]>0)[0] # ancestor of the leaf node
			b1 = []
			while b.shape[0]>0:
				idx = b[0]  # parent index
				b1.insert(0,idx)
				b = np.where(tree_mtx[:,idx]>0)[0] # ancestor of the leaf node
			
			b1 = np.append(b1,leaf_idx)  # with the node as the end
			path_vec.append(b1)

		return path_vec

	def _search_leaf(self):
		tree_mtx = self.tree_mtx
		n1 = tree_mtx.shape[0] # number of nodes
		
		leaf_vec = []
		for i in range(0,n1):
			idx = np.where(tree_mtx[i,:]>0)[0]
			if idx.shape[0]==0:
				leaf_vec.append(i)

		return np.array(leaf_vec)

	def _matrix1(self): 
		
		tree_mtx = self.tree_mtx
		leaf_vec = self.leaf_vec

		print self.branch_dim
		n2, N2 = self.node_num, self.node_num   # assign a branch to the first node
		n1 = np.array(leaf_vec).shape[0]        # the number of leaf nodes 
		N1 = int(n1*(n1-1)/2)

		print N1, N2
		pair_list, parent_list = [], [None]*n2

		A1 = np.zeros((n1,n2))  # leaf node number by branch dim
		print "path_vec", self.path_vec
		for i in range(0,n1):
			print leaf_vec[i], self.path_vec[i]

		for i in range(0,n2):
			b = np.where(tree_mtx[:,i]>0)[0]
			if b.shape[0]>0:
				parent_list[i] = b[0]
			else:
				parent_list[i] = []

		for i in range(0,n1):
			leaf_idx = leaf_vec[i]
			A1[i,parent_list[leaf_idx]] = 1

		A2 = np.zeros((N1,N2))
		cnt = 0
		for i in range(0,n1):
			# leaf_idx1 = leaf_vec[i]
			vec1 = self.path_vec[i]
			for j in range(i+1,n1):
				# leaf_idx2 = leaf_vec[j]
				vec2 = self.path_vec[j]
				t1 = np.intersect1d(vec1,vec2)  # common ancestors
				id1 = np.max(t1)  # the nearest common ancestor

				c1 = np.setdiff1d(vec1, t1)
				c2 = np.setdiff1d(vec2, t1)

				A2[cnt,c1], A2[cnt,c2] = 1, 1
				# common_ans[cnt] = id1
				pair_list.append([leaf_vec[i],leaf_vec[j],id1])

				cnt += 1

		print pair_list
		filename = "ou_A1.txt"
		np.savetxt(filename, A1, fmt='%d', delimiter='\t')
		filename = "ou_A2.txt"
		np.savetxt(filename, A2, fmt='%d', delimiter='\t')

		return A1, A2, pair_list, parent_list

	def _ou_lik_varied(self, params, state_id):

		n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
		
		c = state_id
		values = np.zeros((n2,2))  # expectation and variance
		covar_mtx = np.zeros((n1,n1))

		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		# alpha, sigma, theta1 = params[0:num1], params[num1:2*num1], params1[2*num1:3*num1+1]
		params1 = params[1:]
		beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

		ratio1 = lambda1/(2*beta1)
		values[0,0] = theta1[0]  # mean value of the root node
		values[0,1] = params[0]
		beta1_exp = np.exp(-beta1)
		beta1_exp = np.insert(beta1_exp,0,0)

		# compute the transformation matrix
		A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

		# add a branch to the first node
		beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
		
		# print p_idx
		print beta1
		
		for i in range(1,n2):
			values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
			values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

		# print values
		s1 = np.matmul(A2, beta1)
		idx = pair_list[:,-1]   # index of common ancestor
		s2 = values[idx,1]*np.exp(-s1)
		
		num = pair_list.shape[0]
		leaf_list = self.leaf_list
		for k in range(0,num):
			id1,id2 = pair_list[k,0], pair_list[k,1]
			i,j = leaf_list[id1], leaf_list[id2]
			covar_mtx[i,j] = s2[k]
			covar_mtx[j,i] = covar_mtx[i,j]

		for i in range(0,n1):
			covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
		
		V = covar_mtx.copy()
		theta = theta1[self.leaf_vec]
		mean_values1 = values[self.leaf_vec,0]
		# obsmean = np.outer(self.stats['obs'][c], theta)
		obsmean = np.outer(self.stats['obs'][c], mean_values1)
		# print covar_mtx, theta

		Sn_w = (self.stats['obs*obs.T'][c]
				- obsmean - obsmean.T
				+ np.outer(mean_values1, mean_values1)*self.stats['post'][c])

		n_samples = self.n_samples
		# weights_sum = stats['post'][c]
		lik = self.stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

		self.values = values.copy()
		self.cv_mtx = covar_mtx.copy()

		# print "likelihood", state_id, lik

		return lik

	def _ou_param_varied_constraint(self, params_vec):
	
		n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
		
		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		for c in range(0,self.n_components):

			values = np.zeros((n2,2))	# expectation and variance
			covar_mtx = np.zeros((n1,n1))
		
			# alpha, sigma, theta1 = params[0:num1], params[num1:2*num1], params1[2*num1:3*num1+1]
			params = params_vec[c,:]
			params1 = params[1:]
			beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

			# b = np.where((lambda1<=1e-07&beta1<=1e-07)==True)[0]
			b = np.where(beta1>1e-07)[0]
			ratio1 = np.zeros(num1)
			ratio1[b] = lambda1[b]/(2*beta1[b])
			values[0,0] = theta1[0]  # mean value of the root node
			values[0,1] = params[0]
			beta1_exp = np.exp(-beta1)
			beta1_exp = np.insert(beta1_exp,0,0)

			# compute the transformation matrix
			A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list

			# add a branch to the first node
			beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
		
			for i in range(1,n2):
				values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
				values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

			# print values
			s1 = np.matmul(A2, beta1)
			idx = pair_list[:,-1]   # index of common ancestor
			s2 = values[idx,1]*np.exp(-s1)

			num = pair_list.shape[0]
			leaf_list = self.leaf_list
			for k in range(0,num):
				id1,id2 = pair_list[k,0], pair_list[k,1]
				i,j = leaf_list[id1], leaf_list[id2]
				covar_mtx[i,j] = s2[k]
				covar_mtx[j,i] = covar_mtx[i,j]

			for i in range(0,n1):
				covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
		
			self.means_[c] = values[self.leaf_vec,0].copy()
			self._covars_[c] = covar_mtx.copy()+self.min_covar*np.eye(self.n_features) 
			
		return True

	def _ou_lik_varied_constraint(self, params, state_id):
	
		n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
		
		c = state_id
		# print "state_id: %d"%(state_id), params
		flag = self._check_params(params)
		if flag <= -2:
			# print "state_id: %d"%(state_id), params
			# raise Exception("nan in params")
			params = self.init_ou_params[state_id].copy()
			lik = self._ou_lik_varied_constraint(params, state_id)
			# print "nan in params restart: use initial parameter estimates %s"%(lik)
			print "nan1"

			return lik

		values = np.zeros((n2,2))	# expectation and variance
		covar_mtx = np.zeros((n1,n1))

		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		params1 = params[1:]
		beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

		# b = np.where((lambda1<=1e-07&beta1<=1e-07)==True)[0]
		b = np.where(beta1>1e-07)[0]
		ratio1 = np.zeros(num1)
		ratio1[b] = lambda1[b]/(2*beta1[b])
		values[0,0] = theta1[0]  # mean value of the root node
		values[0,1] = params[0]
		beta1_exp = np.exp(-beta1)
		beta1_exp = np.insert(beta1_exp,0,0)

		# compute the transformation matrix
		A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

		# add a branch to the first node
		beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
		
		for i in range(1,n2):
			values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
			values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

		# print values
		s1 = np.matmul(A2, beta1)
		idx = pair_list[:,-1]   # index of common ancestor
		s2 = values[idx,1]*np.exp(-s1)
		
		num = pair_list.shape[0]
		leaf_list = self.leaf_list
		for k in range(0,num):
			id1,id2 = pair_list[k,0], pair_list[k,1]
			i,j = leaf_list[id1], leaf_list[id2]
			covar_mtx[i,j] = s2[k]
			covar_mtx[j,i] = covar_mtx[i,j]

		for i in range(0,n1):
			covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
		
		# sigma1 = sigma**2
		# V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
		# V = covar_mtx.copy()
		V = covar_mtx.copy()+self.min_covar*np.eye(self.n_features)
		theta = theta1[self.leaf_vec]
		mean_values1 = values[self.leaf_vec,0]
		# obsmean = np.outer(self.stats['obs'][c], theta)
		obsmean = np.outer(self.stats['obs'][c], mean_values1)
		# print covar_mtx, theta

		Sn_w = (self.stats['obs*obs.T'][c]
				- obsmean - obsmean.T
				+ np.outer(mean_values1, mean_values1)*self.stats['post'][c])

		n_samples = self.n_samples
		lambda_0 = self.lambda_0

		lambda_1 = 1.0/np.sqrt(n_samples)

		flag1 = False
		cnt = 0
		cnt1 = 0
		lik = np.nan
		while flag1==False:
			if np.linalg.cond(V) < 1/sys.float_info.epsilon:
				flag1 = True
				lik = (self.stats['post'][c]*np.log(det(V)+small_eps)/n_samples
						+np.sum(inv(V)*Sn_w)/n_samples
						+lambda_0*lambda_1*np.dot(params.T,params))
			else:
				if cnt<10:
					V = V+self.min_covar*np.eye(self.n_features)	# handle it
					cnt = cnt+1
					continue
				else:
					print "matrix not invertible! %d"%(cnt)
					#print V
					print det(V)
					# print params
					# print "state_id: %d"%(state_id), params
					try:
						# V = V+self.min_covar*np.eye(self.n_features)	# handle it
						pinv_V = np.linalg.pinv(V)		# handle it
						eps=1e-12
						lik = (self.stats['post'][c]*np.log(det(V)+small_eps)/n_samples
								+np.sum(pinv_V*Sn_w)/n_samples
								+lambda_0*lambda_1*np.dot(params.T,params))
						# lik = (-self.stats['post'][c]*np.log(det(pinv_V)+eps)/n_samples
						# 		+np.sum(pinv_V*Sn_w)/n_samples
						# 		+lambda_0*lambda_1*np.dot(params.T,params))
						flag1 = True
					except Exception as err:
						#raise
						print("OS error: {0}".format(err))
						break

		self.values = values.copy()
		self.cv_mtx = V.copy()

		# print "likelihood", state_id, lik

		return lik

	# compute log likelihood for a single state
	def _ou_lik_varied_single(self, params, obs):

		n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
		
		values = np.zeros((n2,2))  # expectation and variance
		covar_mtx = np.zeros((n1,n1))

		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		# alpha, sigma, theta1 = params[0:num1], params[num1:2*num1], params1[2*num1:3*num1+1]
		params1 = params[1:]
		beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

		ratio1 = lambda1/(2*beta1)
		values[0,0] = theta1[0]  # mean value of the root node
		values[0,1] = params[0]
		beta1_exp = np.exp(-beta1)
		beta1_exp = np.insert(beta1_exp,0,0)

		# compute the transformation matrix
		A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

		# add a branch to the first node
		beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
		
		for i in range(1,n2):
			values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
			values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

		# print values
		s1 = np.matmul(A2, beta1)
		idx = pair_list[:,-1]   # index of common ancestor
		s2 = values[idx,1]*np.exp(-s1)
		
		num = pair_list.shape[0]
		leaf_list = self.leaf_list
		for k in range(0,num):
			id1,id2 = pair_list[k,0], pair_list[k,1]
			i,j = leaf_list[id1], leaf_list[id2]
			covar_mtx[i,j] = s2[k]
			covar_mtx[j,i] = covar_mtx[i,j]

		for i in range(0,n1):
			covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
		
		# sigma1 = sigma**2
		# V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
		# V = covar_mtx.copy()
		V = covar_mtx.copy()+self.min_covar*np.eye(self.n_features)
		mean_values1 = values[self.leaf_vec,0]
		n = obs.shape[0]
		# obsmean = np.outer(self.stats['obs'][c], theta)
		# obsmean = np.outer(n, mean_values1)
		obsmean = np.outer(np.mean(obs,axis=0),mean_values1)
		# obs1 = obs-mean_values1
		# print covar_mtx
		# print values

		Sn_w = np.dot(obs.T,obs)/n - obsmean - obsmean.T + np.outer(mean_values1, mean_values1)

		# print Sn_w
		# print V

		# weights_sum = stats['post'][c]
		lik = np.log(det(V))+np.sum(inv(V)*Sn_w)

		flag1 = False
		cnt = 0
		cnt1 = 0
		lik = np.nan
		while flag1==False:
			if np.linalg.cond(V) < 1/sys.float_info.epsilon:
				flag1 = True
				lik = np.log(det(V))+np.sum(inv(V)*Sn_w)
			else:
				if cnt<10:
					V = V+self.min_covar*np.eye(self.n_features)	# handle it
					cnt = cnt+1
					continue
				else:
					print "matrix not invertible! %d"%(cnt)
					#print V
					print det(V)
					# print params
					# print "state_id: %d"%(state_id), params
					try:
						# V = V+self.min_covar*np.eye(self.n_features)	# handle it
						pinv_V = np.linalg.pinv(V)		# handle it
						lik = np.log(det(V))+np.sum(inv(V)*Sn_w)
						flag1 = True
					except Exception as err:
						#raise
						print("OS error: {0}".format(err))
						break


		self.values = values.copy()
		self.cv_mtx = covar_mtx.copy() + self.min_covar*np.eye(self.n_features)

		# print "likelihood ou varied single", lik

		return lik

	def _ou_optimize(self, state_id):
		initial_guess = np.random.rand((1,self.n_params))
		# initial_guess = self.params_vec1[state_id].copy()

		method_vec = ['L-BFGS-B','BFGS','SLSQP','COBYLA','Nelder-Mead','Newton-CG']
		id1 = 0
		# con1 = {'type': 'ineq', 'fun': constraint1}
		con1 = {'type':'ineq', 'fun': lambda x: x-1e-07}

		res = minimize(ou_lik,initial_guess,args = (self.cv_mtx, state_id,
						self.stats, self.leaf_time, self.n_samples),
					   constraints=con1, tol=1e-5, options={'disp': False})

		lik = self._ou_lik(res.x, self.cv_mtx, state_id)
		
		return res.x, lik

	def _ou_optimize2(self, state_id):
		
		cnt = 0
		flag = False

		while flag<=0:
			if cnt>0:
				if flag1==-1:
					print "out of bound error! %d"%(cnt)
				else:
					print "NAN error! %d"%(cnt)
			flag, params1 = self._ou_optimize2_unit(state_id)
			flag1 = self._check_params(params1)
			cnt = cnt + 1 
			if cnt>10:
				break
		
		if flag>0 and flag1>0:
			lik = self._ou_lik_varied_constraint(params1, state_id)
		else:
			print "out of bound error! use initial paramter estimates"
			params1 = self.init_ou_params[state_id].copy()
			lik = self._ou_lik_varied_constraint(params1, state_id)
		
		return params1, lik

	def _ou_optimize2_unit(self, state_id):
		
		# initial_guess = 1*np.random.rand(self.n_params)
		a1 = self.initial_w1
		a2 = self.initial_w1a
		w2 = self.initial_w2
		n1 = self.node_num

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
		id1 = 0
		cnt = 0
		flag1 = False
		con1 = ({'type': 'ineq', 'fun': lambda x: x-small_eps},	# selection strength and variance are positive and have upperbounds
				{'type': 'ineq', 'fun': lambda x: -x+100})

		while flag1==False:
			try:
				flag1 = True
				if self.initial_mode==1:
					random1 = 2*np.random.rand(self.n_params)-1
					random1[0:-n1] = np.random.rand(self.n_params-n1)
					random1 = w2*random1
				else:
					random1 = w2*np.random.rand(self.n_params)

				initial_guess = (a1*self.init_ou_params[state_id].copy() 
							+ a2*self.params_vec1[state_id].copy()
							+ (1-a1-a2)*random1)
				print "initial guess", initial_guess
				
				res = minimize(self._ou_lik_varied_constraint, initial_guess, args = (state_id),
							constraints=con1, tol=1e-6, options={'disp': False})
				
			except Exception as err:
				print("OS error: {0}".format(err))
				flag1 = False
				cnt = cnt + 1 
				if cnt > 10:
					print "cannot find the solution! %d"%(cnt)
					break

		if flag1==True:
			params1 = res.x

			flag = self._check_params(params1)

			return flag, params1

		else:

			return False, initial_guess

	def _check_params(self, params):

		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		params1 = params[1:]
		beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

		# need to update the limit values
		min1, max1 = 0, 1e02
		min2, max2 = -1e02, 1e02

		flag1 = (beta1>=min1)&(beta1<=max1)&(lambda1>=min1)&(lambda1<=max1)
		flag2 = (theta1>=min2)&(theta1<=max2)

		flag3 = np.where(np.isnan(params1))[0]

		if sum(flag1)<num1 or sum(flag2)<num1+1:
			if len(flag3)>0:
				return -2
			return -1
		else:
			return 1

	def _ou_optimize_init(self, X, mean_values):

		cnt = 0
		flag = -1

		while flag<=0:
			if cnt>0:
				if flag==-1:
					print "out of bound error! %d"%(cnt)
				else:
					print "NAN error! %d"%(cnt)
			flag, params1 = self._ou_optimize_init_unit(X, mean_values)
			flag = self._check_params(params1)
			cnt = cnt + 1 
			if cnt>20:
				break
		
		if flag>0:
			lik = self._ou_lik_varied_single(params1, X)
		else:
			print "out of bound error! use random initial values"
			params1 = self._ou_init_guess(mean_values)
			lik = self._ou_lik_varied_single(params1, X)
		
		return params1, lik

	def _ou_init_guess(self,mean_values):

		initial_guess = self.initial_w2*np.random.rand(self.n_params)
		
		p_idx = self.parent_list
		leaf_vec = self.leaf_vec

		print "p_idx", p_idx
		print "leaf_vec", leaf_vec
		
		# mean_values1[leaf_vec] = mean_values.copy()
		n2 = leaf_vec.shape[0]

		n1 = self.node_num
		mean_values1 = np.zeros(n1)
		# for i in range(0,n2):
			# p_idx = leaf_vec[i]	# parent of leaf
		
		flag = np.zeros(n1)
		mean_values1[leaf_vec] = mean_values.copy()
		flag[leaf_vec] = 2

		for j in range(n1-1,0,-1):
			p_id1 = p_idx[j]
			if flag[p_id1]==0:
				mean_values1[p_id1] = mean_values1[j]
				flag[p_id1] += 1
			elif flag[p_id1]==1:
				mean_values1[p_id1] = 0.5*mean_values1[p_id1]+0.5*mean_values1[j]
				flag[p_id1] += 1

		initial_guess[self.n_params-n1:self.n_params] = mean_values1.copy() # initialize the mean values

		# print "initial guess", initial_guess

		return initial_guess

	def _ou_optimize_init_unit(self, X, mean_values):

		initial_guess = self._ou_init_guess(mean_values)

		print "initial guess", initial_guess

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
		id1 = 0
		n1 = self.node_num
		
		con1 = ({'type': 'ineq', 'fun': lambda x: x-small_eps},	# selection strength and variance are positive and have upperbounds
				{'type': 'ineq', 'fun': lambda x: -x+100})
		res = minimize(self._ou_lik_varied_single, initial_guess, args = (X),
						constraints=con1, tol=1e-6, options={'disp': False})

		params1 = res.x
		# lik = self._ou_lik_varied_single(params1, X)
		# return params1, lik

		flag = self._check_params(params1)

		return flag, params1

	def _do_mstep(self, stats):
		super(phyloHMRF, self)._do_mstep(stats)

		self.stats = stats.copy()
		means_prior = self.means_prior
		means_weight = self.means_weight

		print "M_step"
		denom = stats['post'][:, np.newaxis]
		#self._output_stats(self.counter)
		#self.counter += 1

		print denom

		if 'c' in self.params:
			print "flag: true covariance"

			for c in range(self.n_components):
				print "state_id: %d"%(c)

				params, value = self._ou_optimize2(c)
				print params
				print value
				self.lik = value
				self.params_vec1[c] = params.copy()

				mean_values = self.values[self.leaf_vec,0]
				self.means_[c] = mean_values.copy()
				self._covars_[c] = self.cv_mtx.copy()+self.min_covar*np.eye(self.n_features) 

		print self.params_vec1
		print self.means_
		print self._covars_
		for c in range(self.n_components):
			print("%.4f\t")%(det(self._covars_[c]))
		print("\n")

		# print self.startprob_
		# print self.transmat_

	def _compute_covariance_mtx_varied(self, params):
		n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
		values = np.zeros((n2,2))	# expectation and variance
		covar_mtx = np.zeros((n1,n1))
		print "leaf number, node number", n1, n2

		num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
		# alpha, sigma, theta1 = params[0:num1], params[num1:2*num1], params1[2*num1:3*num1+1]
		params1 = params[1:]
		beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

		ratio1 = lambda1/(2*beta1)

		values[0,0] = theta1[0]  # mean value of the root node
		values[0,1] = params[0]
		beta1_exp = np.exp(-beta1)
		beta1_exp = np.insert(beta1_exp,0,0)

		# compute the transformation matrix
		A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

		# add a branch to the first node
		beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
		
		# print p_idx
		print beta1
		# print beta1_exp
		print lambda1
		print theta1
		# print ratio1
		
		for i in range(1,n2):
			values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
			values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

		# print values
		s1 = np.matmul(A2, beta1)
		idx = pair_list[:,-1]   # index of common ancestor
		s2 = values[idx,1]*np.exp(-s1)
		
		num = pair_list.shape[0]
		leaf_list = self.leaf_list
		 
		for k in range(0,num):
			id1,id2 = pair_list[k,0], pair_list[k,1]
			i,j = leaf_list[id1], leaf_list[id2]
			covar_mtx[i,j] = s2[k]
			covar_mtx[j,i] = covar_mtx[i,j]

		for i in range(0,n1):
			covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance

		mean_values1 = values[self.leaf_vec,0]
		return covar_mtx.copy(), mean_values1.copy()

	def _load_simu_parameters(self, filename1):
		simu_params = scipy.io.loadmat(filename1)

		# start_prob, transmat, branch_param = simu_params['startprob'], simu_params['transmat'], simu_params['branch_param']
		start_prob, equili_prob, transmat, branch_param = simu_params['startprob'], simu_params['equiliprob'], simu_params['transmat'], simu_params['branch_param']
		print branch_param.shape
		# print start_prob, transmat

		n_components, n_components1 = np.array(start_prob).shape[0], np.array(branch_param).shape[0]    # the number of states
		
		if n_components!=n_components1:
			print "the number of components is error!"

		covar_mtx = np.zeros((n_components,self.n_features,self.n_features))
		mean_values = np.zeros((n_components,self.n_features))
		print covar_mtx.shape
		for i in range(0,n_components):
			covar_mtx[i], mean_values[i] = self._compute_covariance_mtx_varied(branch_param[i])
			#print temp1
			#print temp1.shape
			print covar_mtx[i]

		return n_components, start_prob, equili_prob, transmat, mean_values, covar_mtx

	def _simu_paramters(n_samples, filename1):
		n_components2, equili_prob, transmat, mean_values, covar_mtx = self._load_simu_parameters(filename1)

	def _check_1(self):
		"""Validates model parameters prior to fitting.

		Raises
		------

		ValueError
			If any of the parameters are invalid, e.g. if :attr:`startprob_`
			don't sum to 1.
		"""
		self.startprob_ = np.asarray(self.startprob_)
		if len(self.startprob_) != self.n_components:
			raise ValueError("startprob_ must have length n_components")
		if not np.allclose(self.startprob_.sum(), 1.0):
			raise ValueError("startprob_ must sum to 1.0 (got {0:.4f})"
							 .format(self.startprob_.sum()))

		self.transmat_ = np.asarray(self.transmat_)
		if self.transmat_.shape != (self.n_components, self.n_components):
			raise ValueError(
				"transmat_ must have shape (n_components, n_components)")
		if not np.allclose(self.transmat_.sum(axis=1), 1.0):
			#raise ValueError("rows of transmat_ must sum to 1.0 (got {0})"
			#                 .format(self.transmat_.sum(axis=1)))
			v1 = self.transmat_.sum(axis=1)
			print "rows of transmat_ must sum to 1.0", v1
			b = np.where(v1<1e-07)[0]
			for id1 in b:
				t1 = np.zeros(self.n_components)
				t1[id1] = 1
				self.transmat_[id1] = t1.copy()

def parse_args():
	parser = OptionParser(usage="Phylo-HMRF state estimation", add_help_option=False)
	parser.add_option("-n", "--num_states", default="8", help="Set the number of states to estimate for HMM model")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-l","--length", default="one", help="Filename of length vectors")
	parser.add_option("-p","--root_path", default=".", help="Root directory of the input data files")
	parser.add_option("-m","--multiple", default="true", help="Use multivariate data (true, default) or single variate data (false) ")
	parser.add_option("-a","--species_name", default="human", help="Species to estimate states (used under single variate mode)")
	parser.add_option("-o","--sort_states", default="false", help="Whether to sort the states")
	parser.add_option("-r","--run_id", default="0", help="experiment id")
	parser.add_option("-c","--cons_param", default="1", help="constraint parameter")
	parser.add_option("-d","--initial_mode", default="0", help="initial mode: 0: positive random vector; 1: positive random vector for branches")
	parser.add_option("-i","--initial_weight", default="0.2", help="initial weight 0 for initial parameters")
	parser.add_option("-k","--initial_weight1", default="0", help="initial weight 1 for initial parameters")
	parser.add_option("-j","--initial_magnitude", default="1", help="initial magnitude for initial parameters")
	parser.add_option("-s","--simu_version", default="1", help="dataset version")
	parser.add_option("-u","--position1", default="902205", help="position1")
	parser.add_option("-v","--position2", default="36541194", help="position2")
	parser.add_option("-w","--filter_sigma", default="0.25", help="sigma of filter")
	parser.add_option("-b","--beta", default="1", help="beta")
	parser.add_option("--beta1",default="0.1",help="beta1")
	parser.add_option("--num_neighbor",default="8",help="number of neighbors")
	parser.add_option("--filter_mode",default="0",help="filter method")
	parser.add_option("-e","--threshold", default="0.001", help="convergence threshold")
	parser.add_option("-g","--estimate_type",default="3",help="choice to consider edge weights: 0: not consider edge weights; 3: consider edge weights")
	parser.add_option("-q","--annotation",default="a1",help="annotation of the filename")

	(opts, args) = parser.parse_args()
	return opts

def run(num_states,chromosome,length_vec,root_path,multiple,species_name,
		sort_states,run_id1,cons_param,
		initial_mode,initial_weight,initial_weight1,initial_magnitude, 
		position1, position2, filter_sigma, beta, beta1, num_neighbor, filter_mode, 
		conv_threshold, estimate_type, simu_version, annotation):
	
	learning_rate=0.001
	run_id = int(run_id1)
	n_components1 = int(num_states)
	cons_param = float(cons_param)
	simu_version = int(simu_version)
	initial_mode = int(initial_mode)
	initial_weight = float(initial_weight)
	initial_weight1 = float(initial_weight1)
	initial_magnitude = float(initial_magnitude)
	version = int(simu_version)
	region_start = int(position1)
	region_stop = int(position2)
	beta = float(beta)
	beta1 = float(beta1)
	num_neighbor = int(num_neighbor)
	conv_threshold = float(conv_threshold)
	estimate_type = int(estimate_type)
	annotation = str(annotation)
	filter_mode = int(filter_mode)

	print "estimate type %d"%(estimate_type)

	# load the edge list
	# data_path = "."	# the directory where phylogeny information files are kept
	data_path = str(root_path)
	filename2 = "%s/edge.1.txt"%(data_path)
	if(os.path.exists(filename2)==True):
		f = open(filename2, 'r')
		print("edge list loaded")
		edge_list = [map(int,line.split('\t')) for line in f]
		print edge_list

	# load branch length file if provided
	filename2 = "%s/branch_length.1.txt"%(data_path)
	if(os.path.exists(filename2)==True):
		f = open(filename2, 'r')
		print("branch list loaded")
		branch_list = [map(float,line.split('\t')) for line in f]
		branch_list = branch_list[0]
		print branch_list

	# load species name
	filename2 = "%s/species_name.1.txt"%(data_path)
	if(os.path.exists(filename2)==True):
		f = open(filename2, 'r')
		print("species names loaded")
		species = [line.strip() for line in f]
		print species[0]

	# chromosome name
	chrom = str(chromosome)
	# chrom = "1"
	resolution = 50000	# the bin size is set to be 50Kb as default

	# load region_list
	filename2 = "%s/chr%s.synteny.txt"%(data_path,chrom)
	if(os.path.exists(filename2)==True):
		t_lenvec = np.loadtxt(filename2, dtype='int', delimiter='\t')	# load *.txt file
		region_list = []
		temp1 = np.ravel(t_lenvec)
		if len(temp1)==3:
			region_list.append(t_lenvec[0:2])
		else:
			num1 = len(t_lenvec)
			for i in range(0,3):
				region_list.append(t_lenvec[i,0:2])

		print region_list

	# load region_list
	filename2 = "%s/path_list.txt"%(data_path)
	if(os.path.exists(filename2)==True):
		f = open(filename2, 'r')
		print("path list loaded")
		filename_list = [line.strip() for line in f]
		print filename_list

	output_path = "."	# the directory where multi-species Hi-C contact file is output
	annot1 = 'observed'
	output_filename = "%s/chr%s.%dKb.%s.txt"%(output_path,chrom,int(resolution/1000),annot1)
	print output_filename

	# write contact frequency of different species into an array (n_samples*(n_position+n_species))
	ref_species = 'hg38'
	type_id = 0
	utility1.multi_contact_matrix(chrom, resolution, filename_list, species, output_filename)

	species_num = 4

	# filename1 = filename
	filename1 = output_filename
	data_ori = pd.read_table(filename1)	# load DataFrame file
	colnames = list(data_ori)

	position = np.asarray(data_ori.loc[:,colnames[0:3]])
	x1 = np.asarray(data_ori.loc[:,colnames[3:]])
	print x1.shape
	print x1[0:10]

	x_min, x_max = 0, -1
	x1, vec1, x_min, x_max = utility1.normalize_feature(x1,x_min,x_max)
	print np.max(x1,axis=0), np.min(x1,axis=0), np.std(x1,axis=0)
	print vec1

	x = np.log(1+x1)

	# num_neighbor = 8
	print "num_neighbor: %d"%(num_neighbor)
	sigma = float(filter_sigma)
	type_id = 0

	edge_list_vec = []
	
	samples = []
	len_vec = []
	id1, id2 = 0, 0
	n_samples_accumulate = 0
	filter_param1, filter_param2 = -1, -1
	if filter_mode==0:
		filter_param1, filter_param2 = 5, 50

	for t_position in region_list:

		print "select regions..."

		position1, position2 = t_position[0], t_position[1] 
		output_filename3 = "%s/chr%s.%dKb.select2.%s.%d.%d.R5.txt"%(output_path,chrom,int(resolution/1000),annot1,position1,position2)
		#output_filename3 = ""
		x1, idx = utility1.select_valuesPosition1(position, x, output_filename3, position1, position2, resolution)

		print position1, position2, x1.shape
		print np.mean(x1,axis=0)
		print np.max(x1,axis=0)

		dim1,dim2 = x1.shape[0], x1.shape[-1]
		for k in range(0,dim2):
			b2 = np.where(x1[:,k]>1e-05)[0]
			print k, dim1, len(b2), np.mean(x1[:,k]), np.median(x1[:,k]), np.max(x1[:,k]), np.mean(x1[b2,k]), np.median(x1[b2,k])			

		t_position = position[idx,:]

		annot2 = 'chr%s.local.test5.graphcut.%s'%(chrom,annotation)
		output_filename1 = "data1_mtx.test.%s.%d.%d.txt"%(annot2,position1,position2)
		output_filename2 = "%s/chr%s.%dKb.edgeList.%s.%d.%d.txt"%(output_path,chrom,int(resolution/1000),annot2,position1,position2)
		#output_filename1 = ""
		#output_filename2 = ""
		#x1, mtx1, t_position, edge_list_1 = utility1.write_matrix_image_Ctrl_v1(x1,t_position,
		#						output_filename1,output_filename2,num_neighbor,sigma,type_id,filter_mode)

		# if the file already exists, it will not be overwritten
		if(os.path.exists(output_filename1)==True):
			output_filename1 = ""
		if(os.path.exists(output_filename2)==True):
			output_filename2 = ""

		x1, mtx1, t_position, edge_list_1 = utility1.write_matrix_image_Ctrl_v2(x1,t_position,output_filename1,output_filename2,
										num_neighbor,sigma,type_id,filter_mode,filter_param1,filter_param2)

		print x1.shape,mtx1.shape,t_position,len(edge_list_1)
		print np.mean(x1,axis=0)
		print np.max(x1,axis=0)

		dim1,dim2 = x1.shape[0], x1.shape[-1]
		for k in range(0,dim2):
			b2 = np.where(x1[:,k]>1e-05)[0]
			print k, dim1, len(b2), np.mean(x1[:,k]), np.median(x1[:,k]), np.max(x1[:,k]), np.mean(x1[b2,k]), np.median(x1[b2,k])			

		samples.extend(x1)
		start_region = np.min(t_position)
		n_samples = x1.shape[0]
		id2 = id1 + n_samples
		n_samples_accumulate = n_samples_accumulate+n_samples

		len_vec.append([n_samples,id1,id2,mtx1.shape[0],mtx1.shape[1],start_region])

		id1 = id2
		
		edge_list_1 = np.asarray(edge_list_1)
		edge_list_1 = np.int64(edge_list_1)
		edge_list_vec.append(edge_list_1)

		dim = mtx1.shape
		print "mtx1:", dim
		dim1, dim2 = dim[0], dim[1]

	samples = np.asarray(samples)
	print samples.shape
	print len_vec
	dim = mtx1.shape
	print dim
	dim1, dim2 = dim[0], dim[1]

	## write samples and region annotations to a dictionary
	# mdict = {}
	# mdict['samples'] = samples
	# mdict['lenvec'] = len_vec
	# filename2 = "%s/%s.%dKb.%s.mat"%(output_path,chrom,int(resolution/1000),annot1)
	# scipy.io.savemat(filename2,mdict)

	start = time.time()

	path_1 = "."
	annot = "phylo-hmrf"
	
	if not os.path.exists(path_1):
		try:
			os.makedirs(path_1)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise  

	sym_Idx = utility1.symmetric_idx(dim1,dim2) 

	tree1 = phyloHMRF(n_components=n_components1, run_id=run_id, n_samples = n_samples_accumulate, n_features = samples[0].shape[-1], 
				n_dim1 = dim[0], n_dim2 = dim[1], 
				observation=samples, observation_mtx=mtx1, edge_list=edge_list, len_vec = len_vec, type_id=version, branch_list=branch_list, edge_list_1 = edge_list_vec, 
				cons_param=cons_param, beta = beta, beta1 = beta1, initial_mode = initial_mode, 
				initial_weight = initial_weight, initial_weight1 = initial_weight1, initial_magnitude = initial_magnitude, 
				learning_rate=learning_rate, estimate_type = estimate_type, max_iter = 100, n_iter=5000, tol=1e-7)

	threshold = conv_threshold
	print threshold

	params_vec, params_vec1, params_vecList, state_vecList, iter_id1, iter_id2, cost_vec = tree1.fit_accumulate(samples, len_vec, threshold)

	lambda_0 = cons_param

	print "predicting states..."
		
	annot2 = 'chr%s.graphcut.joint.%d.%d.%s'%(chrom,position1,position2,annotation)

	path_1 = "./"

	# save the estimated states and parameters
	filename3 = "%s/estimate_ou_%d_%.2f_%d_s%d.%s.txt"%(path_1, run_id, lambda_0, n_components1, version, annot2)

	mdict = {}
	mdict['params_vecList'] = params_vecList
	mdict['state_vecList'] = state_vecList
	mdict['params_vec'], mdict['params_vec1'], mdict['iter_id1'], mdict['iter_id2'], mdict['cost_vec'] = params_vec, params_vec1, iter_id1, iter_id2, cost_vec
	filename3 = "%s/estimate_ou_%d_%.2f_%d_s%d.%s.mat"%(path_1, run_id, lambda_0, n_components1, version, annot2)
	scipy.io.savemat(filename3,mdict)
	print params_vecList.shape

	end = time.time()
	print "use time:"
	print(end - start)
	
if __name__ == '__main__':

	opts = parse_args()
	run(opts.num_states,opts.chromosome,opts.length,opts.root_path,opts.multiple,\
		opts.species_name,opts.sort_states,opts.run_id,opts.cons_param, \
		opts.initial_mode, opts.initial_weight, opts.initial_weight1, opts.initial_magnitude, \
		opts.position1, opts.position2, opts.filter_sigma, opts.beta, opts.beta1, opts.num_neighbor, \
		opts.filter_mode, opts.threshold, opts.estimate_type, \
		opts.simu_version, opts.annotation)
