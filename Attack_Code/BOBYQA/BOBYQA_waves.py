# coding: utf-8

# # BOBYQA Attack Procedure

# With this file we aim to generate the function that allows to attack a given image tensor

from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
# import scipy.misc
# from numba import jit
# import math
import time

# Bobyqa
import pybobyqa
import pandas as pd
import scipy.fftpack as fft

# import matplotlib.pyplot as plt

# Initialisation Coefficients

MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be


class Objfun(object):
	def __init__(self, objfun):
		self._objfun = objfun
		self.nf = 0
		self.xs = []
		self.fs = []

	def __call__(self, x):
		self.nf += 1
		self.xs.append(x.copy())
		f = self._objfun(x)
		self.fs.append(f)
		return f

	def get_summary(self, with_xs=False):
		results = {}
		if with_xs:
			results['xvals'] = self.xs
		results['fvals'] = self.fs
		results['nf'] = self.nf
		results['neval'] = np.arange(1, self.nf + 1)  # start from 1
		return pd.DataFrame.from_dict(results)

	def reset(self):
		self.nf = 0
		self.xs = []
		self.fs = []

def attribution(X,indx,c):
	# c jas to be in the nxnx3 shape already
	n,_,_ = c.shape
	for i in range(len(indx)):
		i_in = indx[i]
		for j in range(len(indx)):
			j_in = indx[j]
			for k in range(3):
				X[i_in,j_in,k] = c[i,j,k]
	return X

def vec2mod(c, indice, var):
	# returns the tensor whose element in indice are var + c
	temp = var.copy()

	n = len(indice)
	#     print(indice)
	for i in range(n):
		temp.reshape(-1)[indice[i]] += c[i]
	return temp

def wave2mod(c, N, idx, var):
	# returns the tensor whose element in indice are var + c
	# print(c)
	n = int(np.sqrt(len(c)/3))
	N = int(np.sqrt(N/3))
	X = np.zeros((N,N,3))
	idx = idx.astype(int)
	c = c.reshape((n,n,3))
	# X[idx,:,:][:,idx,:] = c.reshape(X[idx,:,:][:,idx,:].shape)
	X = attribution(X,idx,c)
	# print(idx)
	# print('Selected X',X[idx,:,:][:,idx,:])
	# print('X',np.max(X))
	# print('X',np.min(X))
	Y = np.zeros(X.shape)
	for i in range(3):
		Y[:,:,i] = fft.idctn(X[:,:,i], norm = 'ortho')
	Y = Y.reshape(-1)
	# print(np.max(Y))
	# print(np.min(Y))
	temp = var.copy()
	for i in range(len(Y)):
		temp.reshape(-1)[i] += Y[i]
	return temp

def clipping(x,a,b):

	# Function with which we cast the function again into its limits
	temp = np.minimum(x.copy().reshape(-1),b)
	temp = np.maximum(temp,a)
	temp = temp.reshape(x.shape)
	# for i in range(len(temp.reshape(-1))):
	# 	if temp.reshape(-1)[i]>b[i]:
	# 		temp.reshape(-1)[i] = b[i]
	# 	elif temp.reshape(-1)[i]<a[i]:
	# 		temp.reshape(-1)[i] = a[i]
	return temp

def sign_max_exp(X,a,b):
	temp = X.copy()
	n = len(temp.reshape(-1))

	temp = np.maximum(temp.copy().reshape(-1), b)
	temp = np.minimum(temp, a)
	temp = temp.reshape(X.shape)

	# for i in range(n):
	# 	if temp.reshape(-1)[i]>b.reshape(-1)[i]:
	# 		temp.reshape(-1)[i] = b.reshape(-1)[i]
	# 	elif temp.reshape(-1)[i]<a.reshape(-1)[i]:
	# 		temp.reshape(-1)[i] = a.reshape(-1)[i]

	return temp


class BlackBox_BOBYQA:
	def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
				 targeted=TARGETED, max_iterations=MAX_ITERATIONS,
				 print_every=1, early_stop_iters=0,
				 abort_early=ABORT_EARLY,
				 use_log=False, use_tanh=True, use_resize=False,
				 start_iter=0, L_inf=0.15,
				 init_size=8, use_importance=True, rank=2,
				 ordered_domain=False, image_distribution=False,
				 mixed_distributions=False, Sampling=True, Rank=False,
				 Rand_Matr=False, Rank_1_Mixed=False, GenAttack=False,
				 max_eval=1e5):
		"""
		The BOBYQA attack.

		Returns adversarial examples for the supplied model.

		confidence: Confidence of adversarial examples: higher produces examples
		  that are farther away, but more strongly classified as adversarial.
		batch_size: Number of gradient evaluations to run simultaneously.
		targeted: True if we should perform a targetted attack, False otherwise.
		binary_search_steps: The number of times we perform binary search to
		  find the optimal tradeoff-constant between distance and confidence.
		max_iterations: The maximum number of iterations. Larger values are more
		  accurate; setting too small will require a large learning rate and will
		  produce poor results.
		abort_early: If true, allows early aborts if gradient descent gets stuck.
		"""

		image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
		self.model = model
		self.sess = sess
		self.rank = rank
		self.TARGETED = targeted
		self.target = 0
		self.Generations = int(1e5)
		self.MAX_ITERATIONS = max_iterations
		self.print_every = print_every
		self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
		self.ABORT_EARLY = abort_early
		self.CONFIDENCE = confidence
		self.start_iter = start_iter
		self.batch_size = batch_size
		self.num_channels = num_channels
		self.use_importance = use_importance
		self.ordered_domain = ordered_domain
		self.image_distribution = image_distribution
		self.mixed_distributions = mixed_distributions

		self.small_x = image_size
		self.small_y = image_size

		self.L_inf = L_inf
		self.use_tanh = use_tanh

		self.max_eval = max_eval

		self.Sampling = Sampling
		self.Rank = Rank
		self.Rand_Matr = Rand_Matr

		# each batch has a different modifier value (see below) to evaluate
		single_shape = (image_size, image_size, num_channels)
		small_single_shape = (self.small_x, self.small_y, num_channels)

		# the variable we're going to optimize over
		# support multiple batches
		self.modifier = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
		self.scaled_modifier = self.modifier
		# the real variable, initialized to 0
		self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)

		# these are variables to be more efficient in sending data to tf
		# we only work on 1 image at once; the batch is for evaluation loss at different modifiers
		self.timg = tf.Variable(np.zeros(single_shape), dtype=tf.float32)
		self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)

		# and here's what we use to assign them
		self.assign_timg = tf.placeholder(tf.float32, single_shape)
		self.assign_tlab = tf.placeholder(tf.float32, num_labels)

		# the resulting image, tanh'd to keep bounded from -0.5 to 0.5
		# broadcast self.timg to every dimension of modifier
		if use_tanh:
			self.newimg = tf.tanh(self.scaled_modifier + self.timg) / 2
		else:
			self.newimg = self.scaled_modifier + self.timg

		# prediction BEFORE-SOFTMAX of the model
		# now we have output at #batch_size different modifiers
		# the output should have shape (batch_size, num_labels)
		self.output = model.predict(self.newimg)
		# compute the probability of the label class versus the maximum other
		# self.tlab * self.output selects the Z value of real class
		# because self.tlab is an one-hot vector
		# the reduce_sum removes extra zeros, now get a vector of size #batch_size
		self.real = tf.reduce_sum((self.tlab) * self.output, 1)

		# (1-self.tlab)*self.output gets all Z values for other classes
		# Because soft Z values are negative, it is possible that all Z values are less than 0
		# and we mistakenly select the real class as the max. So we minus 10000 for real class
		self.other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)
		self.use_max = False
		self.sum = tf.reduce_sum((1 - self.tlab) * self.output, 1)
		# self.logsum  = tf.reduce_sum(tf.log((1-self.tlab)*self.output- (self.tlab*10000)),1)
		# _,index_ord  = tf.nn.top_k(self.tlab*self.output,10)
		# self.temp    = tf.gather(self.output,tf.cast(index_ord[0],tf.int32))
		# self.k_max   = tf.reduce_sum(self.temp,0)
		# If self.targeted is true, then the targets represents the target labels.
		# If self.targeted is false, then targets are the original class labels.
		if self.TARGETED:
			# The loss is log(1 + other/real) if use log is true, max(other - real) otherwise
			self.loss = tf.log(tf.divide(self.sum + 1e-30, self.real + 1e-30))
			# self.loss_max = tf.log(tf.divide(self.other +1e-30,self.real+1e-30))
			self.distance = tf.maximum(0.0, self.other - self.real + self.CONFIDENCE)

		# these are the variables to initialize when we run
		self.setup = []
		self.setup.append(self.timg.assign(self.assign_timg))
		self.setup.append(self.tlab.assign(self.assign_tlab))

		# prepare the list of all valid variables
		var_size = self.small_x * self.small_y * num_channels
		self.use_var_len = var_size
		self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
		self.used_var_list = np.zeros(var_size, dtype=np.int32)
		self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

		# upper and lower bounds for the modifier
		self.image_size = image_size
		self.num_channels = num_channels
		self.var_size_b = image_size * image_size * num_channels
		self.modifier_up = np.zeros(self.var_size_b, dtype=np.float32)
		self.modifier_down = np.zeros(self.var_size_b, dtype=np.float32)

	def blackbox_optimizer_ordered_domain(self, iteration, ord_domain):
		# build new inputs, based on current variable value
		var = self.real_modifier.copy()
		var_size = self.real_modifier.size

		NN = self.var_list.size
		nn = self.batch_size

		# We choose the elements of ord_domain that are inerent to the step. So it is already
		# limited to the variable's dimension
		# if ((iteration + 1) * nn <= NN):
		# 	var_indice = ord_domain[iteration * nn: (iteration + 1) * nn]
		# else:
		# 	var_indice = ord_domain[
		# 		list(range(iteration * nn, NN)) + list(range(0, (self.batch_size - (NN - iteration * nn))))]
		if ((iteration + 1) * nn <= NN):
			var_indice = ord_domain[-(iteration + 1) * nn:-iteration * nn]
		else:
			var_indice = ord_domain[
				list(range(iteration * nn, NN)) + list(range(0, (self.batch_size - (NN - iteration * nn))))]

		b = self.modifier_up
		a = self.modifier_down

		opt_fun = Objfun(lambda c: self.sess.run([self.loss], feed_dict={
			self.modifier: clipping(wave2mod(c, NN, var_indice, var),a,b)})[0])

		x_o = np.zeros((np.power(self.batch_size,2)*3,))
		# Checking if the function evaluation changes real_modifier
		if not (var - self.real_modifier).any() == 0:
			print('Not the same after the evalluation')
			print(var.shape, type(var))
			print(self.real_modifier, type(self.real_modifier))

		initial_loss = opt_fun(x_o)
		# print(initial_loss)
		aa = -np.ones(x_o.shape)
		bb = np.ones(x_o.shape)

		soln = pybobyqa.solve(opt_fun, x_o, rhobeg=np.min(bb - aa) / 3,
							  bounds=(aa, bb), maxfun=nn**2*3 * 4,
							  rhoend=np.min(bb - aa) / 6,
							  npt=nn**2*3 + 1)

		summary = opt_fun.get_summary(with_xs=False)
		# TO DO: get the difference vlaue
		evaluations = soln.nf + 1
		# adjust sample probability, sample around the points with large gradient
		# print('Within Optimisation',soln.f)
		nimgs = clipping(wave2mod(soln.x, NN, var_indice, var),a,b)
		distance = self.sess.run(self.distance, feed_dict={self.modifier: nimgs})

		return distance[0], evaluations + 1, nimgs, summary

	def attack(self, imgs, targets):
		"""
		Perform the L_2 attack on the given images for the given targets.

		If self.targeted is true, then the targets represents the target labels.
		If self.targeted is false, then targets are the original class labels.
		"""
		r = []
		self.target = np.argmax(targets)
		print('The target is', self.target)
		print('go up to', len(imgs))
		# we can only run 1 image at a time, minibatches are used for gradient evaluation
		for i in range(0, len(imgs)):
			print('tick', i)
			r.extend(self.attack_batch(imgs[i], targets[i]))
		return np.array(r)

	def attack_batch(self, img, lab):
		"""
		Run the attack on a batch of images and labels.
		"""
		self.target = np.argmax(lab)

		def compare(x, y):
			if not isinstance(x, (float, int, np.int64)):
				x = np.copy(x)
				if self.TARGETED:
					x[y] -= self.CONFIDENCE
				else:
					x[y] += self.CONFIDENCE
				x = np.argmax(x)
			if self.TARGETED:
				return x == y
			else:
				return x != y

		# remove the extra batch dimension
		if len(img.shape) == 4:
			img = img[0]
		if len(lab.shape) == 2:
			lab = lab[0]

		# set the lower and upper bounds accordingly
		lower_bound = 0.0
		upper_bound = 1e10

		# convert img to float32 to avoid numba error
		img = img.astype(np.float32)
		img0 = img

		# clear the modifier
		self.real_modifier.fill(0.0)

		# the best l2, score, and image attack
		o_bestl2 = 1e10
		o_bestscore = -1
		o_bestattack = img
		eval_costs = 0

		if self.ordered_domain:
			print(np.random.choice(10, 3))
			ord_domain = np.random.choice(self.var_list.size, self.var_list.size, replace=False, p=self.sample_prob)
		else:
			ord_domain = np.arange(np.sqrt(self.var_list.size/self.num_channels))

		print('There are at most ', self.MAX_ITERATIONS, ' iterations.')

		self.sess.run(self.setup, {self.assign_timg: img,
								   self.assign_tlab: lab})

		previous_loss = 1e6
		count_steady = 0
		global_summary = []
		for step in range(self.MAX_ITERATIONS):
			# use the model left by last constant change
			prev = 1e6
			train_timer = 0.0
			last_loss1 = 1.0

			attack_time_step = time.time()

			# print out the losses every 10%
			if step % (self.print_every) == 0:
				loss, output, distance = self.sess.run((self.loss, self.output, self.distance),
													   feed_dict={self.modifier: self.real_modifier})
				print(
					"[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss_f = {:.5g}, maxel = {}".format(
						step, eval_costs, train_timer, self.real_modifier.shape, distance[0], loss[0],
						np.argmax(output[0])))
				sys.stdout.flush()

			attack_begin_time = time.time()
			printing_time_initial_info = attack_begin_time - attack_time_step

			zz = np.zeros(img.shape)
			ub = 0.5 * np.ones(img.shape)
			lb = -ub

			if not self.use_tanh:  # and (step>0):
				scaled_modifier = self.sess.run([self.scaled_modifier], feed_dict={self.modifier: self.real_modifier})[
					0]

				if step == 0:
					scaled_modifier = img

				self.modifier_up = np.maximum(np.minimum(- (img.reshape(-1, ) - img0.reshape(-1, )) + self.L_inf,
														 ub.reshape(-1, ) - img.reshape(-1, ))
											  , zz.reshape(-1, ))
				self.modifier_down = np.minimum(np.maximum(- (img.reshape(-1, ) - img0.reshape(-1, )) - self.L_inf,
														   - img.reshape(-1, ) + lb.reshape(-1, ))
												, zz.reshape(-1, ))

			if step > 0:
				self.sess.run(self.setup, {self.assign_timg: img,
										   self.assign_tlab: lab})
				self.real_modifier.fill(0.0)


			l, evaluations, nimg, summary = self.blackbox_optimizer_ordered_domain(step, ord_domain)
			global_summary.append(summary)

			self.real_modifier = nimg
			if (previous_loss - l) < 1e-3:
				count_steady += 1

			previous_loss = l

			adv = self.sess.run([self.scaled_modifier], feed_dict={self.modifier: self.real_modifier})[0]

			temp = np.minimum(self.modifier_up.reshape(-1, ),
							  adv.reshape(-1, ))
			adv = temp
			adv = np.maximum(self.modifier_down,
							 adv)
			adv = adv.reshape((self.image_size, self.image_size, self.num_channels))
			img = img + adv
			eval_costs += evaluations

			# check if we should abort search if we're getting nowhere.
			if self.ABORT_EARLY and step % self.early_stop_iters == 0:
				if l > prev * .9999:
					print("Early stopping because there is no improvement")
					return o_bestattack, eval_costs
				prev = l

			# Find the score output
			loss_test, score, real, other = self.sess.run((self.loss, self.output, self.real, self.other),
														  feed_dict={self.modifier: nimg})

			score = score[0]

			# adjust the best result found so far
			# the best attack should have the target class with the largest value,
			# and has smallest l2 distance

			if l < o_bestl2 and compare(score, np.argmax(lab)):
				# print a message if it is the first attack found
				if o_bestl2 == 1e10:
					print(
						"[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}".format(
							step, eval_costs, train_timer, self.real_modifier.shape, l))
					sys.stdout.flush()
				o_bestl2 = l
				o_bestscore = np.argmax(score)
				o_bestattack = img

			# If loss if 0 return the result
			if eval_costs > self.max_eval:
				print('The algorithm did not converge')
				return o_bestattack, eval_costs, global_summary

			if l <= 0:
				print("Early Stopping becuase minimum reached")
				return o_bestattack, eval_costs, global_summary


		return o_bestattack, eval_costs, global_summary# coding: utf-8
