# coding: utf-8

# # BOBYQA Attack Procedure

# With this file we aim to generate the function that allows to attack a given image tensor

from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import scipy.misc
from numba import jit
import math
import time

#Bobyqa
import pybobyqa
import matplotlib.pyplot as plt

# Initialisation Coefficients

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be


def vec2mod(c,indice,var):
    # returns the tensor whose element in indice are var + c
    temp = var.copy()
   
    n = len(indice)
#     print(indice)
    for i in range(n):
        temp.reshape(-1)[indice[i]] += c[i]
    return temp

def generator_rank1_img(c,img):
    # this function returns an image tensor given the ordered vector; the 
    # first component is the rowa, the second the columns, and the third
    # the colour pixels (if RGB this is three)

    var = img.copy()
    _,n,m,r = var.shape

    for k in range(r):
        for i in range(n):
            for j in range(m):
#                 print(c[i])
                var[0,i,j,k] = c[i]*c[j+n]*c[k+n+m]

    return var

def generator_rankn_img(c,img,rank):
    # this function returns an image tensor given the ordered vector; the 
    # first component is the rowa, the second the columns, and the third
    # the colour pixels (if RGB this is three)

    var = img.copy()
    _,n,m,r = var.shape
    
    nn = n+m+r
    
    for rr in range(int(rank)):
        for k in range(r):
            for i in range(n):
                for j in range(m):
                    var[0,i,j,k] += c[i+nn*rr]*c[j+n+nn*rr]*c[k+n+m+nn*rr]
                
    return var

def generator_rand_gaus_img(c,img,n,M):
    # this function returns an image tensor given the ordered vector; the 
    # first component is the rowa, the second the columns, and the third
    # the colour pixels (if RGB this is three)

    var = img.copy()
    
    for i in range(n):
        var[0,:,:,:] += c[i]*M[i]
                
    return var

def summation_rand_uni_img(c,img,n,M):
    # this function returns an image tensor given the ordered vector; the 
    # first component is the rowa, the second the columns, and the third
    # the colour pixels (if RGB this is three)
    var = img.copy()
    
    for i in range(n):
        var[0,:,:,:] += c[i]*M[i]
                
    return var

def boundaries(a_img,b_img,x_size,y_size,chan_size):
    
    a = np.zeros(x_size+y_size+chan_size,)
    b = np.zeros(x_size+y_size+chan_size,)
    
    for i in range(x_size):
        temp_a = []#np.zeros(y_size*chan_size,)
        temp_b = []#np.zeros(y_size*chan_size,)
        for j in range(y_size):
            for k in range(chan_size):
                temp_a.append(float(a_img[i,j,k]))
                temp_b.append(float(b_img[i,j,k]))
        a[i] = -np.power(abs(max(temp_a)),1/3)
        b[i] = np.power(min(temp_b),1/3)
        
    for j in range(y_size):
        temp_a = []#np.zeros(y_size*chan_size,)
        temp_b = []#np.zeros(y_size*chan_size,)
        for i in range(x_size):
            for k in range(chan_size):
                temp_a.append(float(a_img[i,j,k]))
                temp_b.append(float(b_img[i,j,k]))
                
        a[j + x_size] = -np.power(abs(max(temp_a)),1/3)
        b[j + x_size] = np.power(min(temp_b),1/3)

    for k in range(chan_size):
        temp_a = []#np.zeros(y_size*chan_size,)
        temp_b = []#np.zeros(y_size*chan_size,)
        for i in range(x_size):
            for j in range(y_size):
                temp_a.append(float(a_img[i,j,k]))
                temp_b.append(float(b_img[i,j,k]))
                
        a[k + x_size + y_size] = -np.power(abs(max(temp_a)),1/3)
        b[k + x_size + y_size] = np.power(min(temp_b),1/3)
        
    return a,b

def boundaries_2(a_img,b_img,x_size,y_size,chan_size):
    # Differently from the previous, here we consider to have just one
    # channel. This allows to devide the boundaries into two. The one 
    # for the second dimension is [-1,1] while for the first one is 
    # [-delta, delta] with delta being the smallest value that the pixel 
    # can achieve
    a = np.zeros(x_size+y_size+chan_size,)
    b = np.zeros(x_size+y_size+chan_size,)
    
    for i in range(x_size):
        temp_a = []#np.zeros(y_size*chan_size,)
        temp_b = []#np.zeros(y_size*chan_size,)
        for j in range(y_size):
            for k in range(chan_size):
                temp_a.append(float(a_img[i,j,k]))
                temp_b.append(float(b_img[i,j,k]))
        a[i] = -abs(max(temp_a))
        b[i] = min(temp_b)
        
    for j in range(y_size):
        temp_a = []#np.zeros(y_size*chan_size,)
        temp_b = []#np.zeros(y_size*chan_size,)
        for i in range(x_size):
            for k in range(chan_size):
                temp_a.append(float(a_img[i,j,k]))
                temp_b.append(float(b_img[i,j,k]))
                
        a[j + x_size] = -1
        b[j + x_size] = 1

    for k in range(chan_size):                
        a[k + x_size + y_size] = -1
        b[k + x_size + y_size] = 1
        
    return a,b
    
def boundaries_random_matrix(a_img,b_img,n,m,p,rank,M):
    # This function computes the boundaries for the random matrix approach.
    bound = np.zeros(rank,)
    
    for r in range(rank):
        a_max  = np.max(a_img)
        b_min  = np.min(b_img)
        M_max  = np.max(abs(M[r]))
        
        bound[r] = min([-a_max,b_min])/M_max
        
    return -bound,bound
        
    
def CrossOver(Parent1,Parent2,F1,F2):
    p = F1/(F1+F2)
    
    choice = np.random.binomial(1,p,1)
    
    return choice*Parent1 + (1-choice)*Parent2

#########################################################
# Functions related to the optimal sampling of an image #
#########################################################
        
def find_neighbours(r,c,k,n,m,R):
    # This computes the neihgbours of a pixels (r,c,k) in an image R^(n,m,R)
    # Note: We never consider differnt layers of the RGB
    neighbours = []
    #print(r,c,k,n,m,R)
    for i in range(-1,2):
        for j in range(-1,2):
    #        print(i,j)
    #        print(((r+i)>=0)and((r+i)<n))
            if (((r+i)>=0)and((r+i)<n)):
                if (((c+j)>=0)and((c+j)<m)):
                    if not((i,j)==(0,0)):
                        neighbours.append([0,r+i,c+j,k])
    #print(neighbours)
    return neighbours
    
def get_variation(img,neighbours):
    list_val = []
#     print('list neighbours',neighbours)
    for i in range(len(neighbours)):
        list_val.append(img[neighbours[i][1]][neighbours[i][2]][neighbours[i][3]])
    #print(list_val)
    sum_var = np.std(list_val) 
    return(sum_var)

    
def total_img_var(row,col,k,img):
    
    n,m,RGB = img.shape
    neighbours = find_neighbours(row,col,k,n,m,RGB)
    total_var = get_variation(img,neighbours)    
    return total_var
    
    
def image_region_importance(img):
    # This function obtaines the image as an imput and it computes the importance of the 
    # different regions. 
    # 
    # Inputs:
    # - img: tensor of the image that we are considering
    # Outputs:
    # - probablity_matrix: matrix with the ecorresponding probabilities.
    
    n,m,k = img.shape
    #print(type(img))
   
    probability_matrix = np.zeros((n,m,k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row,col,i] = total_img_var(row,col,i,img)
#                 print(probability_matrix[row,col,i])
    
    # We have to give a probability also to all the elelemnts that have zero variance
    # this implies that we will add to the whole matrix the minimum nonzero value, divided
    # by 100
    probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/100
    
    # Normalise the probability matrix
    probability_matrix = probability_matrix/np.sum(probability_matrix)
    
    return probability_matrix


class BlackBox_BOBYQA:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, max_iterations = MAX_ITERATIONS, 
                 print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY,
                 use_log = False, use_tanh = True, use_resize = False,
                 start_iter = 0, L_inf = 0.15,
                 init_size = 8, use_importance = True, rank = 2, 
                 ordered_domain = False, image_distribution = False,
                 mixed_distributions = False, Sampling = True, Rank = False,
                 Rand_Matr = False,Rank_1_Mixed  = False, GenAttack = False,
                 max_eval = 1e5):
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
        self.resize_init_size = init_size
        self.use_importance = use_importance
        self.ordered_domain = ordered_domain
        self.image_distribution = image_distribution
        self.mixed_distributions = mixed_distributions
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size
        
        self.L_inf      = L_inf
        self.use_tanh   = use_tanh
        self.use_resize = use_resize
        
        self.use_log    = tf.placeholder(tf.bool)
        self.use_log2   = use_log

        self.max_eval   = max_eval
        
        if (Rank) or (Rand_Matr):
            self.Sampling = False
            if Rank and Rand_Matr:
                print('It is not possible to do both rank and rand Matrix methods')
                return -1
            else:
                self.Rank = Rank
                self.Rand_Matr = Rand_Matr
        else:
            self.Sampling = Sampling
            self.Rank = Rank
            self.Rand_Matr = Rand_Matr
            
        if Rank_1_Mixed :
            self.Sampling = False
            self.Rank = False
            self.Rand_Matr = False
            self.Rank_1_Mixed = True
            
        self.GenAttack = GenAttack
        if GenAttack :
            self.Sampling = False
            self.Rank = False
            self.Rand_Matr = False
            self.Rank_1_Mixed = False
        
        # each batch has a different modifier value (see below) to evaluate
        shape = (None,image_size,image_size,num_channels)
        single_shape = (image_size, image_size, num_channels)
        small_single_shape = (self.small_x, self.small_y, num_channels)

        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size
        if self.use_resize:
            self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
            # scaled up image
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_size, image_size],method=tf.image.ResizeMethod.BILINEAR,align_corners=True)
            # operator used for resizing image
            self.resize_size_x = tf.placeholder(tf.int32)
            self.resize_size_y = tf.placeholder(tf.int32)
            self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
            self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y],align_corners=True,method=tf.image.ResizeMethod.BILINEAR)
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            # no resize
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
            self.newimg = tf.tanh(self.scaled_modifier + self.timg)/2
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
        self.real = tf.reduce_sum( (self.tlab)*self.output ,1)#(tf.reduce_sum( (self.tlab)*self.output ,1)- tf.reduce_min(self.output))/tf.reduce_sum(self.output- tf.reduce_min(self.output)) 
        
        # (1-self.tlab)*self.output gets all Z values for other classes
        # Because soft Z values are negative, it is possible that all Z values are less than 0
        # and we mistakenly select the real class as the max. So we minus 10000 for real class
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)#(tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000)- tf.reduce_min(self.output),1)) / tf.reduce_sum(self.output - tf.reduce_min(self.output)) 

        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            # The loss is log(1 + other/real) if use log is true, max(other - real) otherwise
            self.loss = tf.cond(tf.equal(self.use_log, tf.constant(True)), lambda: tf.log( 1 + tf.divide(self.other ,self.real)), lambda: tf.maximum(0.0, self.other-self.real+self.CONFIDENCE))
#             if self.use_log:
# #                 temp_loss = np.min(self.output)
# #                 temp_out  = self.output-temp_loss
# #                 self.real = tf.reduce_sum((self.tlab)*temp_out,1)
# #                 self.other = tf.reduce_max((1-self.tlab)*temp_out - (self.tlab*10000),1)

#                 loss1 = tf.log(self.other + 1e-0) - tf.log(self.real + 1e-0)
# #                 loss1 = tf.maximum(0.0,  tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
#                 self.loss_Gen = tf.log( 1 + tf.divide(self.other ,self.real))
#                 loss1 = self.loss_Gen
#                 if not tf.is_finite(loss1):
#                     loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
#             else:
#                 # if targetted, optimize for making the other class (real) most likely
#                 loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
#                 self.loss_Gen = self.other-self.real+self.CONFIDENCE


#         else:
#             if self.use_log:
#                 loss1 = tf.maximum(0.0, tf.log(self.real + 1e-16) - tf.log(self.other + 1e-16))
#             else:
#                 # if untargeted, optimize for making this class least likely.
#                 loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)

        # sum up the losses (output is a vector of #batch_size)
#         self.loss = loss1
        

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.used_var_list = np.zeros(var_size, dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size

        # upper and lower bounds for the modifier
        self.image_size = image_size
        self.num_channels = num_channels
        self.var_size_b = image_size * image_size * num_channels
        self.modifier_up = np.zeros(self.var_size_b, dtype = np.float32)
        self.modifier_down = np.zeros(self.var_size_b, dtype = np.float32)
        
    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i+size, j:j+size] = np.max(image[i:i+size, j:j+size])
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double = False):
        prev_modifier = np.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0]*2, old_shape[1]*2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype = np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:,:,i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                prob[:,:,i] = scipy.misc.imresize(image_pool, 2.0, 'nearest', mode = 'F')
            else:
                prob[:,:,i] = image_pool
        prob /= np.sum(prob)
        return prob


    def resize_img(self, small_x, small_y, reset_only = False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(self.real_modifier)
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x, self.resize_size_y: self.small_y, self.resize_input: self.real_modifier})
            print(self.real_modifier.shape)
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        
    
    
    def blackbox_optimizer_Uniform_Matrices(self,iteration):
        # build new inputs, based on current variable value
        dim = self.small_x
        n   = self.rank
        var = self.real_modifier.copy()
        var_size = self.real_modifier.size
        
        M   = []
        for i in range(n):
            M.append(np.random.rand(self.small_x ,self.small_y ,self.num_channels) - np.ones((self.small_x ,self.small_y ,self.num_channels))/2)
        
        opt_fun = lambda c: self.sess.run([self.loss],feed_dict = {self.use_log: self.use_log2,self.modifier: summation_rand_unif_img(c,var,n,M,!!!!!!boundsaries!!!!!)})[0]
        
        # self.modifier_up[]
        x_o = np.zeros(n,)

        print('Highest Modif ', max(self.modifier_up))
        print('Lowest Modifi ', min(self.modifier_down))
        
        b_img = self.modifier_up.reshape(self.small_x,self.small_y,
                                         self.num_channels)/n
        a_img = self.modifier_down.reshape(self.small_x,self.small_y,
                                         self.num_channels)/n
        
        a,b   = boundaries_random_matrix(a_img,b_img,self.small_x,self.small_y,
                              self.num_channels,n,M)
                
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg = 0.1, 
                          bounds = (a,b),maxfun=n*10,
                          npt=n*2)
        print(soln)
        evaluations = soln.nf
        # adjust sample probability, sample around the points with large gradient
        nimgs = generator_rand_gaus_img(soln.x,var,n,M)
        
        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)

        return soln.f, evaluations, nimgs
    
    def blackbox_optimizer_Random_Matrices(self,iteration):
        # build new inputs, based on current variable value
        dim = self.small_x
        n   = self.rank
        var = self.real_modifier.copy()
        var_size = self.real_modifier.size
        
        M   = []
        for i in range(n):
            M.append(2*np.random.binomial(1,1/2,(self.small_x ,self.small_y ,self.num_channels)) - np.ones((self.small_x ,self.small_y ,self.num_channels)))      #np.random.randn(self.small_x ,self.small_y ,self.num_channels)/np.sqrt(self.small_x))
        
        opt_fun = lambda c: self.sess.run([self.loss],feed_dict = {self.use_log: self.use_log2,self.modifier: generator_rand_unif_img(c,var,n,M)})[0]
        
        # self.modifier_up[]
        x_o = np.zeros(n,)

        print('Highest Modif ', max(self.modifier_up))
        print('Lowest Modifi ', min(self.modifier_down))
        
        b_img = self.modifier_up.reshape(self.small_x,self.small_y,
                                         self.num_channels)/n
        
        a_img = self.modifier_down.reshape(self.small_x,self.small_y,
                                         self.num_channels)/n
        
        a,b   = boundaries_random_matrix(a_img,b_img,self.small_x,self.small_y,
                              self.num_channels,n,M)
                
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg = 0.1, 
                          bounds = (a,b),maxfun=n*10,
                          npt=n*2)
        print(soln)
        evaluations = soln.nf
        # adjust sample probability, sample around the points with large gradient
        nimgs = generator_rand_gaus_img(soln.x,var,n,M)
        
        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)

        return soln.f, evaluations, nimgs
    
        
    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        self.target = np.argmax(targets)
        print('The target is', self.target)
        print('go up to',len(imgs))
        # we can only run 1 image at a time, minibatches are used for gradient evaluation
        for i in range(0,len(imgs)):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i], targets[i]))
        return np.array(r)

    def attack_batch(self, img, lab):
        """
        Run the attack on a batch of images and labels.
        """
        self.target = np.argmax(lab)
        def compare(x,y):
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
        
        if self.mixed_distributions & self.image_distribution:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        
        if self.Sampling & self.Rank & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Sampling & self.Rank:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Sampling & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Rank & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        
        if len(img.shape) == 4:
            img = img[0]
        if len(lab.shape) == 2:
            lab = lab[0]
        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        upper_bound = 1e10

        # convert img to float32 to avoid numba error
        img  = img.astype(np.float32)
        img0 = img
        
        # set the upper and lower bounds for the modifier
        # if not self.use_tanh:
        #     self.modifier_up = 0*img.reshape(-1) + self.L_inf
        #     self.modifier_down = 0*img.reshape(-1) - self.L_inf
                
        # clear the modifier
        resize_iter = 0
        if self.use_resize:
            self.resize_img(self.resize_init_size, self.resize_init_size, True)
        else:
            self.real_modifier.fill(0.0)

        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = img
        eval_costs = 0
        
        if self.ordered_domain:
            print(np.random.choice(10,3))
            ord_domain = np.random.choice(self.var_list.size, self.var_list.size, replace=False, p = self.sample_prob)
            
        started_with_log_now_normal = False
        
        iteration_scale = -1
        print('There are at most ',self.MAX_ITERATIONS,' iterations.')
        
        self.sess.run(self.setup, {self.assign_timg: img,
                           self.assign_tlab: lab})
        
        previous_loss = 1e6
        count_steady  = 0
        for step in range(self.MAX_ITERATIONS):
            
            # set the variables so that we don't have to send them over again
            # self.sess.run(self.setup, {self.assign_timg: img,
            #                            self.assign_tlab: lab})
            # self.real_modifier.fill(0.0)
            # use the model left by last constant change
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
#             if self.use_resize:
#                 self.resize_img(self.resize_init_size, self.resize_init_size, True)
#             else:
#                 self.real_modifier.fill(0.0)

#             if self.use_resize:
#                 if step >= self.MAX_ITERATIONS//3:
#                     self.resize_img(64,64)
#                 if step >= 2*(self.MAX_ITERATIONS//3):
#                     self.resize_img(128,128)

            # print out the losses every 10%
            if step%(self.print_every) == 0:
                loss, output = self.sess.run((self.loss,self.output), feed_dict={self.use_log : self.use_log2,self.modifier: self.real_modifier})
                print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}".format(step, eval_costs, train_timer, self.real_modifier.shape, loss[0]))
                sys.stdout.flush()
            
            if started_with_log_now_normal and (step>10000):
                self.use_log2 = True
                started_with_log_now_noraml = False
                print('We allow use_log to be true again')
                l = self.sess.run((self.loss),feed_dict={self.use_log: self.use_log2,self.modifier: self.real_modifier})
                print('Temporary Loss',l)

            if not np.isfinite(loss):
                self.use_log2 = False
                started_with_log_now_normal = True
                print('We no impose use_log to be false')
                loss,output = self.sess.run((self.loss,self.output),feed_dict={self.use_log: self.use_log2,self.modifier:self.real_modifier})
                print('Temporary Loss',loss)
               
            attack_begin_time = time.time()
            # perform the attack
            
            zz = np.zeros(img.shape)
            ub = 0.5*np.ones(img.shape)
            lb = -ub
            
            if self.use_resize:
                zz = np.zeros((self.image_size,self.image_size,self.num_channels))
                ub = 0.5*np.ones((self.image_size,self.image_size,self.num_channels))
                lb = -ub
            
            if not self.use_tanh:# and (step>0):
                scaled_modifier = self.sess.run([self.scaled_modifier],feed_dict = {self.use_log: self.use_log2, self.modifier: self.real_modifier})[0]
                
                
                if step == 0:
                    scaled_modifier = img#tf.image.resize_images(img, [self.small_x, self.small_x]).eval()                    
#                 print('Scaled modifier has minimum', np.min(scaled_modifier),' and maximum ', np.max(scaled_modifier))
#                 print('Scaled_mod has shape',scaled_modifier.shape)
            
#                 # print('The variable nimg has dimension', nimg.size)   

                # print('#### Analysing the diffferent bounds ###############')
                # print('scaled modifier is', scaled_modifier,'with max',np.max(scaled_modifier),' and min', np.min(scaled_modifier))
                # print('scaled modifier - img is', scaled_modifier - img,'with max',np.max(scaled_modifier-img),' and min', np.min(scaled_modifier-img))
                # print('scaled modifier - img  + L_ing is', scaled_modifier - img + self.L_inf,'with max',np.max(scaled_modifier-img + self.L_inf),' and min', np.min(scaled_modifier-img + self.L_inf))
                # print('ub - img', ub - scaled_modifier,'with max',np.max(ub - scaled_modifier),' and min', np.min(ub - scaled_modifier))
                # # print('max (scaled modifier - img  + L_ing, ub -img) is', scaled_modifier - img + self.L_inf,'with max',np.max(scaled_modifier-img + self.L_inf),' and min', np.min(scaled_modifier-img + self.L_inf))
                # print('img - lb',  scaled_modifier - lb ,'with max',np.max(scaled_modifier -lb),' and min', np.min(scaled_modifier - lb ))
                
                self.modifier_up   = np.maximum( np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.L_inf,
                                                           ub.reshape(-1,) - img.reshape(-1,))
                                                ,zz.reshape(-1,))
                self.modifier_down = np.minimum( np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.L_inf,
                                                           - img.reshape(-1,) + lb.reshape(-1,) )
                                                ,zz.reshape(-1,))
                # print('The gaps in the space are,', self.modifier_up-self.modifier_down)
                # print('Low is', self.modifier_down)
                # print('Up is', self.modifier_up)
                # print('and the smallest is: ', np.min(np.abs(self.modifier_up-self.modifier_down)))
            
            
            if step>0:
                self.sess.run(self.setup, {self.assign_timg: img,
                                           self.assign_tlab: lab})
                self.real_modifier.fill(0.0)
            
            if self.Sampling:
                if self.ordered_domain:  
                    
                    iteration_scale += 1
                    iteration_domain = np.mod(iteration_scale,(self.use_var_len//self.batch_size + 1))
                    
                    force_renew = False
                    if iteration_domain == 0:#( iteration_scale >= (self.use_var_len//self.batch_size + 1)):
                        # We force to regenerate a distribution of the nodes if we have 
                        # already optimised over all of the domain
                        force_renew = True
                        
                    if self.use_resize:
                        # Note that we change the dimension of the domain according 
                        # to the step that we are in, not the iteration_scale
                        if (count_steady == 10):
                            previous_loss =  1e6
                            count_steady  =  0
                            resize_iter   += 1
                        
                        if step == 15:#30:#(resize_iter == 1) and (previous_loss == 1e6):#
                            self.small_x = 33
                            self.small_y = 33
                            self.resize_img(self.small_y, self.small_x, False)
                            iteration_scale = 0
                            iteration_domain = 0
                            force_renew = True

                        # if step == 50:#140:#(resize_iter == 2) and (previous_loss == 1e6):#step == 100:
                        #     # print('Now we will use dimension 64')
                        #     self.small_x = 33
                        #     self.small_y = 33
                        #     self.resize_img(self.small_y, self.small_x, False)
                        #     iteration_scale = 0
                        #     iteration_domain = 0
                        #     force_renew = True
                        
                        if step == 50:#400:#(resize_iter == 3) and (previous_loss == 1e6):#step == 200:
                            # print('Now we will use dimension 64')
                            self.small_x = 67
                            self.small_y = 67
                            self.resize_img(self.small_y, self.small_x, False)
                            iteration_scale = 0
                            iteration_domain = 0
                            force_renew = True
                            
#                         if step == 430:#(resize_iter == 1) and (previous_loss == 1e6):#
#                             self.small_x = 17
#                             self.small_y = 17
#                             self.resize_img(self.small_y, self.small_x, False)
#                             iteration_scale = 0
#                             iteration_domain = 0
#                             force_renew = True

#                         if step == 500:#(resize_iter == 2) and (previous_loss == 1e6):#step == 100:
#                             # print('Now we will use dimension 64')
#                             self.small_x = 33
#                             self.small_y = 33
#                             self.resize_img(self.small_y, self.small_x, False)
#                             iteration_scale = 0
#                             iteration_domain = 0
#                             force_renew = True
                        
#                         if step == 650:#(resize_iter == 3) and (previous_loss == 1e6):#step == 200:
#                             # print('Now we will use dimension 64')
#                             self.small_x = 67
#                             self.small_y = 67
#                             self.resize_img(self.small_y, self.small_x, False)
#                             iteration_scale = 0
#                             iteration_domain = 0
#                             force_renew = True
                            
                        if step == 100:#(resize_iter == 4) and (previous_loss == 1e6):#step == 600:
                            # print('Now we will use dimension 64')
                            self.small_x = 135
                            self.small_y = 135
                            self.resize_img(self.small_y, self.small_x, False)
                            iteration_scale = 0
                            iteration_domain = 0
                            force_renew = True

                        if step == 150:#2000:#(resize_iter == 5) and (previous_loss == 1e6):#step == 1200:
                            self.small_x = 299
                            self.small_y = 299
                            self.resize_img(self.small_y, self.small_x, False)
                            iteration_scale = 0
                            iteration_domain = 0
                            force_renew = True
                            
                    
                    if (np.mod(iteration_scale,30)==0) or force_renew:#(iteration == 0):
                        # We have to repermute the pixels if they are modifie
                    
                        if self.image_distribution:
                                                                                    
                            if self.use_resize:
                                prob = image_region_importance(tf.image.resize_images(img,[self.small_x,self.small_y],align_corners=True,method=tf.image.ResizeMethod.BILINEAR).eval()).reshape(-1,)
                            else:
                                prob = image_region_importance(img).reshape(-1,)
                                # prob = np.log(prob)-np.min(np.log(prob))+1e-4
                                # prob = (prob) / (np.sum(prob))
                            # print('The image is of dimension: ',self.real_modifier.shape)
                            # print('Prob has length: ',len(prob))
                            # print('Boh')
                            # print('The self.var_len is: ', self.use_var_len)
                            ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = prob)
                        elif self.mixed_distributions:
                            if (step == 0):
                                prob = image_region_importance(img).reshape(-1,)
                                ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = prob)
                            else:
                                ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = self.sample_prob)
                        else:
                            ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = self.sample_prob)
                        
                        
                        # plt.hist(prob,bins = 100)
                        # print(prob)
                    # print('at step ',step,' the self.small_x is', self.small_x)
                    # print('at step ',step,' the dimension of variable self.use_var_len',self.use_var_len)
                    # print('at step ',step,' the dimension of the constraints is ',len(self.modifier_up))
                    # print(ord_domain,iteration_domain)
                    l, evaluations, nimg = self.blackbox_optimizer_ordered_domain(iteration_domain,ord_domain)
                else:
                    # Normal perturbation method
                    l, evaluations, nimg = self.blackbox_optimizer(step)
            
            elif self.Rank:
                l, evaluations, nimg = self.blackbox_optimizer_Rank_1(step)
                if l == -1:
                    return o_bestattack,eval_costs
            elif self.Rand_Matr:
                l, evaluations, nimg = self.blackbox_optimizer_Random_Matrix(step)
#             print(l)
            
            elif self.Rank_1_Mixed:
                if (step == 0):
                    l, evaluations, nimg = self.blackbox_optimizer_Rank_1(step)
                else:
                    prob = image_region_importance(self.real_modifier[0]).reshape(-1,)
                    ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = prob)
                    l, evaluations, nimg = self.blackbox_optimizer_ordered_domain(step,ord_domain)
            elif self.GenAttack:
                l, evaluations, nimg = self.blackbox_optimizer_GEN_ATTACK(step)
                if l == -4:
                    print('The algorithm did not converge')
                    return nimg, evaluations
            
            
            self.real_modifier = nimg
            if (previous_loss - l)<1e-5:
                count_steady += 1
                
            previous_loss = l
            
            adv = self.sess.run([self.scaled_modifier],feed_dict = {self.use_log: self.use_log2, self.modifier: self.real_modifier})[0]
            # print(type(self.modifier_up.reshape(-1,)))
            # print(type(adv))
            # print(len(adv.reshape(-1,)))
            # print(len(self.modifier_up.reshape(-1,)))
            # print(adv)
            temp = np.minimum(self.modifier_up.reshape(-1,),
                         adv.reshape(-1,))
            adv  = temp
            adv = np.maximum(self.modifier_down,
                         adv)
            adv = adv.reshape((self.image_size,self.image_size,self.num_channels))
            # adv[] = 0.5
            # adv[] = -0.5
            # print('######## NEW ITERATION ################')
            # print('The max bound is:', np.max(self.modifier_up))
            # print('The min bound is:', np.min(self.modifier_down))
            # print('The max bound with image is:', np.max(img + self.modifier_up.reshape((self.image_size,self.image_size,self.num_channels))))
            # print('The min bound with image is:', np.min(img + self.modifier_down.reshape((self.image_size,self.image_size,self.num_channels))))
            img = img + adv
#             print('nu has an absolute maximum of:', np.max(np.abs(adv)))
            # if step%(self.print_every) == 0:
            #     print('####### Importnat checks ########')
            #     print('maximum distance: ',np.max(np.abs(img-img0)))
            #     print('The image has a maximum of:', np.max(img))
            #     print('The image has a minimum of:', np.min(img))
            #     print('#######################################')

            # plt.imshow(adv.reshape((299,299,3))+0.5)
            # eval_costs += evaluations
            
            # check if we should abort search if we're getting nowhere.
            if self.ABORT_EARLY and step % self.early_stop_iters == 0:
                if l > prev*.9999:
                    print("Early stopping because there is no improvement")
                    return o_bestattack, eval_costs
                prev = l

            # Find the score output
            loss_test, score, real, other = self.sess.run((self.loss,self.output,self.real,self.other), feed_dict={self.use_log: self.use_log2,self.modifier: nimg})
            
#             print('test',loss_test)
#             print(l)
#             print('the loss is',l,'and the target value is ',real,' and the classified one is',other,' with the score being',score)
    
            score = score[0]
            
#             if abs(loss_test-l)>1e-6:
#                 print('The score is not relative to the real image')
#                 break
            
            # adjust the best result found so far
            # the best attack should have the target class with the largest value,
            # and has smallest l2 distance
     
            if l < o_bestl2 and compare(score, np.argmax(lab)):
                # print a message if it is the first attack found
                if o_bestl2 == 1e10:
                    print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}".format(step, eval_costs, train_timer, self.real_modifier.shape, l))
                    sys.stdout.flush()
                o_bestl2 = l
                o_bestscore = np.argmax(score)
                o_bestattack = img
                
            # If loss if 0 return the result
            if eval_costs > self.max_eval :
                print('The algorithm did not converge')
                return o_bestattack, eval_costs
            
            if self.use_log2:
                if l < np.log(2):
                    print("Early Stopping becuase minimum reached")
                    return o_bestattack, eval_costs
            else:
                if l==0:
                    print("Early Stopping becuase minimum reached")
                    return o_bestattack, eval_costs

            train_timer += time.time() - attack_begin_time

        # return the best solution found
        return o_bestattack, eval_costs