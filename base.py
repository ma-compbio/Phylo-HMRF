from __future__ import print_function

import string
import sys
import os
from collections import deque

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt

import scipy.misc
import scipy.io
import multiprocessing as mp

import time

#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))

class ConvergenceMonitor(object):
	"""Monitors and reports convergence to :data:`sys.stderr`.

	Parameters
	----------
	tol : double
		Convergence threshold. EM has converged either if the maximum
		number of iterations is reached or the log probability
		improvement between the two consecutive iterations is less
		than threshold.

	n_iter : int
		Maximum number of iterations to perform.

	verbose : bool
		If ``True`` then per-iteration convergence reports are printed,
		otherwise the monitor is mute.

	Attributes
	----------
	history : deque
		The log probability of the data for the last two training
		iterations. If the values are not strictly increasing, the
		model did not converge.

	iter : int
		Number of iterations performed while training the model.
	"""
	_template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

	def __init__(self, tol, n_iter, verbose):
		self.tol = tol
		self.n_iter = n_iter
		self.verbose = verbose
		self.history = deque(maxlen=2)
		self.iter = 0

	def __repr__(self):
		class_name = self.__class__.__name__
		params = dict(vars(self), history=list(self.history))
		return "{0}({1})".format(
			class_name, _pprint(params, offset=len(class_name)))

	def report(self, logprob):
		"""Reports convergence to :data:`sys.stderr`.

		The output consists of three columns: iteration number, log
		probability of the data at the current iteration and convergence
		rate.  At the first iteration convergence rate is unknown and
		is thus denoted by NaN.

		Parameters
		----------
		logprob : float
			The log probability of the data as computed by EM algorithm
			in the current iteration.
		"""
		if self.verbose:
			delta = logprob - self.history[-1] if self.history else np.nan
			message = self._template.format(
				iter=self.iter + 1, logprob=logprob, delta=delta)
			print(message, file=sys.stderr)

		self.history.append(logprob)
		self.iter += 1

	@property
	def converged(self):
		"""``True`` if the EM algorithm converged and ``False`` otherwise."""
		# XXX we might want to check that ``logprob`` is non-decreasing.
		return (self.iter == self.n_iter or
				(len(self.history) == 2 and
				 self.history[1] - self.history[0] < self.tol))

class _BaseGraph(BaseEstimator):
	"""Base class for Markov Random Field Models.
	"""
	def __init__(self, n_components=1, run_id=0, estimate_type=0, weight_type=0,
				 startprob_prior=1.0, transmat_prior=1.0,
				 algorithm="viterbi", random_state=None,
				 n_iter=10, tol=1e-2, verbose=False,
				 params=string.ascii_letters,
				 init_params=string.ascii_letters):
		self.n_components = n_components
		self.params = params
		self.init_params = init_params
		self.startprob_prior = startprob_prior
		self.transmat_prior = transmat_prior
		self.algorithm = algorithm
		self.random_state = random_state
		self.n_iter = n_iter
		self.tol = tol
		self.verbose = verbose
		self.run_id = run_id
		self.estimate_type = estimate_type
		self.weight_type = weight_type

	def score_samples(self, X, lengths=None):
		"""Compute the log probability under the model and compute posteriors.
		"""
		# check_is_fitted(self, "startprob_")
		# self._check()

	def predict(self, X, lengths=None):
		"""Find most likely state sequence corresponding to ``X``.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.

		lengths : array-like of integers, shape (n_sequences, ), optional
			Lengths of the individual sequences in ``X``. The sum of
			these should be ``n_samples``.

		Returns
		-------
		state_sequence : array, shape (n_samples, )
			Labels for each sample from ``X``.
		"""
		# _, state_sequence = self.decode(X, lengths)
		#state = self._estimate_state_graphcuts(X)
		#return state

	def predict_proba(self, X, lengths=None):
		"""Compute the posterior probability for each state in the model.

		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.

		lengths : array-like of integers, shape (n_sequences, ), optional
			Lengths of the individual sequences in ``X``. The sum of
			these should be ``n_samples``.

		Returns
		-------
		posteriors : array, shape (n_samples, n_components)
			State-membership probabilities for each sample from ``X``.
		"""
		_, posteriors = self.score_samples(X, lengths)
		return posteriors

	def fit_accumulate(self, X, len_vec, threshold, lengths=None):

		print("Initilization...")
		# X = check_array(X)
		self._init(X, lengths=lengths)
		print("return from initialization...")
		print("starting...")
		self._check()

		print("model fitting...")
		self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
		# self.n_iter = 100
		max_iter = int(self.max_iter)
		max_iter1 = 20	# iterations after the previous minimum
		type_id = 0
		pairwise_cost_pre, unary_cost_pre, cost1_pre = 0.001, 0.001, 0.001
		threshold1, threshold2 = threshold, threshold
		# threshold1, threshold2 = 1e-03, 1e-03
		cost_vec = []
		min_cost = [0,1000]
		min_cost1 = [0,1000]
		params_vec = self.params_vec1.copy()
		params_vec1 = self.params_vec1.copy()
		num_region = len(len_vec)

		ratio_vec = np.zeros(num_region)
		for i in range(0,num_region):
			ratio_vec[i] = len_vec[i][0]

		n_samples = int(sum(ratio_vec))
		ratio_vec = ratio_vec*1.0/n_samples
		print(n_samples)
		
		params_vecList = []
		state_vecList = []
		
		for iter in range(self.n_iter):
			print(iter)
			stats = self._initialize_sufficient_statistics()
			curr_logprob = 0

			self.queue = mp.Queue()

			print("processes")
			start = time.time()
			# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
			processes = [mp.Process(target=self._predict_posteriors, 
						args=(X, len_vec, region_id, self.queue)) for region_id in range(0,num_region)]

			# Run processes
			for p in processes:
				p.start()

			# m_queue.put((region_id, labels, posteriors, t_pairwise_cost1, t_pairwise_cost, t_unary_cost, t_cost1))
			print("query")
			results = [self.queue.get() for p in processes]
			print(len(results))			

			# Exit the completed processes
			print("join")
			for p in processes:
				p.join()

			end = time.time()
			print("use time %d:"%(iter))
			print(end - start)

			pairwise_cost1, pairwise_cost, unary_cost, cost1 = 0, 0, 0, 0
			
			id1 = 3
			labels = np.zeros(n_samples)
			for i in range(0, num_region):		
				vec1 = results[i]
				# print(vec1[1])
				region_id = vec1[0]
				pairwise_cost1 += vec1[id1]*ratio_vec[region_id]
				pairwise_cost += vec1[id1+1]*ratio_vec[region_id]
				unary_cost += vec1[id1+2]*ratio_vec[region_id]
				cost1 += vec1[id1+3]*ratio_vec[region_id]
				s1, s2 = len_vec[region_id][1], len_vec[region_id][2]
				stats = self._accumulate_sufficient_statistics_1(stats, vec1[1])
				labels[s1:s2] = vec1[2]

				print(vec1[id1:id1+4])

			t_difference1 = abs((pairwise_cost-pairwise_cost_pre)*1.0/pairwise_cost_pre)
			t_difference2 = abs((unary_cost-unary_cost_pre)*1.0/unary_cost_pre)
			t_difference3 = abs((cost1-cost1_pre)*1.0/cost1_pre)
			
			# print("Maximization...")
			print(pairwise_cost_pre,pairwise_cost,unary_cost_pre,unary_cost,cost1_pre,cost1)
			print(t_difference1,t_difference2,t_difference3)
			pairwise_cost_pre, unary_cost_pre, cost1_pre = pairwise_cost, unary_cost, cost1
			cost_vec.append([iter, pairwise_cost, unary_cost, cost1])

			params_vecList.append(self.params_vec1.copy())
			state_vecList.append(labels)
			self.labels = labels.copy()

			if cost1<min_cost[1]:
				min_cost = [iter,cost1]
				params_vec = self.params_vec1.copy()
				self.labels_local = self.labels.copy()	# current local optimal state estimate
				
				print("another temp min")

			if cost1<min_cost1[1] and iter>=3:
				min_cost1 = [iter,cost1]
				params_vec1 = self.params_vec1.copy()
				print("another temp min from iteration 3")

			if (t_difference1<threshold1 and t_difference2<threshold2) or (t_difference3<threshold1):
				break

			if iter>max_iter:
				break

			if iter-min_cost1[0]>max_iter1:
				break

			print("Maximization...")

			self._do_mstep(stats)

		self.params_vec1 = params_vec1.copy()
		self._ou_param_varied_constraint(params_vec)

		print(min_cost)
		print(params_vec)
		cost_vec = np.asarray(cost_vec)
		print(cost_vec)

		params_vecList = np.asarray(params_vecList)
		state_vecList = np.asarray(state_vecList)

		return params_vec, params_vec1, params_vecList, state_vecList, min_cost[0], min_cost1[0], cost_vec

	def fit_accumulate_test(self, X, len_vec, threshold, annotation, m_iter, lengths=None):
		"""Estimate model parameters.
		"""
		print("Initilization...")
		# X = check_array(X)
		start = time.time()
		self._init(X, lengths=lengths)
		print("return from initialization...")
		end = time.time()
		print("use time %s:"%(end-start))

		print("starting...")
		self._check()

		print("model fitting...")
		self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
		# self.n_iter = 100
		max_iter = m_iter
		max_iter1 = 50	# iterations after the previous minimum
		type_id = 0
		pairwise_cost_pre, unary_cost_pre, cost1_pre = 0.001, 0.001, 0.001

		threshold1, threshold2 = threshold, threshold
		# threshold1, threshold2 = 1e-03, 1e-03
		cost_vec = []
		min_cost = [0,1000]
		min_cost1 = [0,1000]
		params_vec = self.params_vec1.copy()
		params_vec1 = self.params_vec1.copy()
		num_region = len(len_vec)

		ratio_vec = np.zeros(num_region)
		for i in range(0,num_region):
			ratio_vec[i] = len_vec[i][0]

		n_samples = int(sum(ratio_vec))
		ratio_vec = ratio_vec*1.0/n_samples
		print(n_samples)
		
		params_vecList = []
		state_vecList = []		
		t_labels = np.zeros(n_samples)

		print("n_iter, m_iter: %d %d"%(self.n_iter, max_iter))
		
		# for iter in range(self.n_iter):
		for iter in range(max_iter):
			print(iter)
			stats = self._initialize_sufficient_statistics()
			curr_logprob = 0

			self.queue = mp.Queue()

			print("processes")
			start = time.time()
			# processes = [mp.Process(target=self._compute_posteriors_graph_test, args=(len_vec, X, region_id,self.posteriors_test,self.posteriors_test1,self.queue)) for region_id in range(0,num_region)]
			processes = [mp.Process(target=self._predict_posteriors, 
						args=(X, len_vec, region_id, self.queue)) for region_id in range(0,num_region)]

			# Run processes
			for p in processes:
				p.start()

			# m_queue.put((region_id, labels, posteriors, t_pairwise_cost1, t_pairwise_cost, t_unary_cost, t_cost1))
			print("query")
			results = [self.queue.get() for p in processes]
			print(len(results))			

			# Exit the completed processes
			print("join")
			for p in processes:
				p.join()

			end = time.time()
			print("use time %d:"%(iter))
			print(end - start)

			pairwise_cost1, pairwise_cost, unary_cost, cost1 = 0, 0, 0, 0
			
			id1 = 3
			labels = np.zeros(n_samples)

			start = time.time()
			for i in range(0, num_region):		
				vec1 = results[i]
				# print(vec1[1])
				region_id = vec1[0]
				pairwise_cost1 += vec1[id1]*ratio_vec[region_id]
				pairwise_cost += vec1[id1+1]*ratio_vec[region_id]
				unary_cost += vec1[id1+2]*ratio_vec[region_id]
				cost1 += vec1[id1+3]*ratio_vec[region_id]
				s1, s2 = len_vec[region_id][1], len_vec[region_id][2]
				stats = self._accumulate_sufficient_statistics_1(stats, vec1[1])
				labels[s1:s2] = vec1[2]

				print(vec1[id1:id1+4])

			end = time.time()
			print("accumulate use time %d:"%(iter))
			print(end - start)

			t_difference1 = abs((pairwise_cost-pairwise_cost_pre)*1.0/pairwise_cost_pre)
			t_difference2 = abs((unary_cost-unary_cost_pre)*1.0/unary_cost_pre)
			t_difference3 = abs((cost1-cost1_pre)*1.0/cost1_pre)
			
			# print("Maximization...")
			print(pairwise_cost_pre,pairwise_cost,unary_cost_pre,unary_cost,cost1_pre,cost1)
			print(t_difference1,t_difference2,t_difference3)
			pairwise_cost_pre, unary_cost_pre, cost1_pre = pairwise_cost, unary_cost, cost1
			cost_vec.append([iter, pairwise_cost, unary_cost, cost1])

			params_vecList.append(self.params_vec1.copy())
			# state_vecList.append(labels)
			self.labels = labels.copy()

			if cost1<min_cost[1]:
				min_cost = [iter,cost1]
				params_vec = self.params_vec1.copy()
				self.labels_local = self.labels.copy()	# current local optimal state estimate				
				print("another temp min")

			if cost1<min_cost1[1] and iter>=3:
				min_cost1 = [iter,cost1]
				params_vec1 = self.params_vec1.copy()
				print("another temp min from iteration 3")
				t_labels = self.labels.copy()  # keep the estimated labels

			if ((t_difference1<threshold1 and t_difference2<threshold2) or (t_difference3<threshold1)) and (iter>5):
				break

			if iter>max_iter:
				break

			if iter-min_cost1[0]>max_iter1:
				break

			print("Maximization...")

			start = time.time()
			self._do_mstep(stats)
			end = time.time()
			print("maximization use time %d %s"%(iter,end-start))

		self.params_vec1 = params_vec1.copy()
		self._ou_param_varied_constraint(params_vec)

		print(min_cost)
		print(params_vec)
		cost_vec = np.asarray(cost_vec)
		print(cost_vec)

		params_vecList = np.asarray(params_vecList)
		# state_vecList = np.asarray(state_vecList)

		return params_vec, params_vec1, params_vecList, min_cost[0], min_cost1[0], cost_vec, t_labels

	def _do_func(self, framelogprob):
		"""Estimate objection function value.

		"""

	def _compute_posteriors_graph(self, X, label, logprob, region_id):
		"""Computes per-component posteriors under the model.

		Parameters
		----------
		X : array-like, shape (n_samples in a synteny region, n_features)
			Feature matrix of individual samples.

		label : array-like, shape (n_samples in a syntenty region)
			current estimated states of individual samples.

		logprob : array-like, shape (n_samples in a synteny region)
			log probability of individual samples.

		region_id: int, index of the synteny region

		Returns
		-------
		posteriors : array, shape (n_samples, n_components)
			Log probability of each sample in ``X`` for each of the
			model states.
		"""

	def _estimate_state_graphcuts(self, X):
		"""Estimates states under the model.

		Parameters
		----------

		-------
		states : array, shape (n_samples, 1)
			Estimated estate of each sample in ``X``
		"""

	def _init(self, X, lengths=None):
		"""Initializes model parameters prior to fitting.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.

		lengths : array-like of integers, shape (n_sequences, )
			Lengths of the individual sequences in ``X``. The sum of
			these should be ``n_samples``.
		"""
		init = 1. / self.n_components
		if 's' in self.init_params or not hasattr(self, "startprob_"):
			self.startprob_ = np.full(self.n_components, init)
		if 't' in self.init_params or not hasattr(self, "transmat_"):
			self.transmat_ = np.full((self.n_components, self.n_components),
									 init)

	def _check(self):
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
			raise ValueError("rows of transmat_ must sum to 1.0 (got {0})"
							 .format(self.transmat_.sum(axis=1)))

	def _compute_log_likelihood(self, X):
		"""Computes per-component log probability under the model.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Feature matrix of individual samples.

		Returns
		-------
		logprob : array, shape (n_samples, n_components)
			Log probability of each sample in ``X`` for each of the
			model states.
		"""

	def _generate_sample_from_state(self, state, random_state=None):
		"""Generates a random sample from a given component.

		"""

	# Methods used by self.fit()

	def _initialize_sufficient_statistics(self):
		"""Initializes sufficient statistics required for M-step.

		"""
		stats = {'nobs': 0,
				 'start': np.zeros(self.n_components),
				 'trans': np.zeros((self.n_components, self.n_components))}
		return stats

	def _accumulate_sufficient_statistics_1(self, stats, stats1):
		"""Updates sufficient statistics from a given sample.

		"""
		stats['post'] += stats1['post']
		stats['obs'] += stats1['obs']
		# stats['obs**2'] = np.zeros((self.n_components, self.n_features)
		stats['obs*obs.T'] += stats1['obs*obs.T']

		return stats

	def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
										  posteriors):
		"""Updates sufficient statistics from a given sample.

		"""
		stats['nobs'] += 1

	def _do_mstep(self, stats):
		"""Performs the M-step of EM algorithm.

		"""
		# The ``np.where`` calls guard against updating forbidden states
		# or transitions in e.g. a left-right HMM.
