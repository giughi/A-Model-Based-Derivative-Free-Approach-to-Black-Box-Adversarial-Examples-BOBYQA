# coding: utf-8

# # BOBYQA Attack Procedure

# With this file we aim to generate the function that allows to attack a given image tensor

from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import scipy.misc
# from numba import jit
import math
import time
import pandas as pd
#Bobyqa
import pybobyqa


# Initialisation Coefficients

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be

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
        results['neval'] = np.arange(1, self.nf+1)  # start from 1
        return pd.DataFrame.from_dict(results)

    def reset(self):
        self.nf = 0
        self.xs = []
        self.fs = []


def vec2mod(c,indice,var):
    # returns the tensor whose element in indice are var + c
    temp = var.copy()

    n = len(indice)
#     print(indice)
    for i in range(n):
        temp.reshape(-1)[indice[i]] += c[i]
    return temp

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
    for i in range(-1,2):
        for j in range(-1,2):
            if ((r+i)>=0)&((r+i)<n):
                if ((c+j)>=0)&((c+j)<m):
                    if not((i,j)==(0,0)):
                        neighbours.append([0,r+i,c+j,k])

    return neighbours

def get_variation(img,neighbours):
    list_val = []
#     print('list neighbours',neighbours)
    for i in range(len(neighbours)):
        list_val.append(img[neighbours[i][1]][neighbours[i][2]][neighbours[i][3]])
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

    probability_matrix = np.zeros((n,m,k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row,col,i] = total_img_var(row,col,i,img)
#                 print(probability_matrix[row,col,i])

    probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/10
    probability_matrix = probability_matrix/np.sum(probability_matrix)

    return probability_matrix


class BlackBox(object):
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, max_iterations = MAX_ITERATIONS,
                 print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY,
                 use_log = False, use_resize = False,
                 start_iter = 0, L_inf = 0.15,
                 init_size = 32, use_importance = True,BOBYQA=True, GenAttack = False,
                 max_eval = 1e4, q = None):
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
        self.TARGETED = targeted
        self.target = 0
        self.Generations = 1000
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size

        if q is None:
            self.q = self.batch_size
        else:
            self.q = q
        self.L_inf      = L_inf
        self.use_resize = use_resize

        self.use_log    = tf.placeholder(tf.bool)
        self.use_log2   = use_log

        self.max_eval   = max_eval

        self.BOBYQA = BOBYQA
        self.GenAttack = GenAttack
        if GenAttack :
            self.BOBYQA = False
            
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
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_size, image_size])
            # operator used for resizing image
            self.resize_size_x = tf.placeholder(tf.int32)
            self.resize_size_y = tf.placeholder(tf.int32)
            self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
            self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y])
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
        self.sum  = tf.reduce_sum((1-self.tlab)*self.output,1)

        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            # self.loss = tf.cond(tf.equal(self.use_log, tf.constant(True)), lambda: tf.log( 1 + tf.divide(self.other ,self.real)), lambda: tf.maximum(0.0, self.other-self.real+self.CONFIDENCE))
            self.loss = tf.cond(tf.equal(self.use_log, tf.constant(True)), lambda: tf.log( 1 + tf.divide(self.other ,self.real)), lambda: tf.maximum(0.0, self.other-self.real+self.CONFIDENCE))
            self.distance = tf.cond(tf.equal(self.use_log, tf.constant(True)), lambda: tf.log( 1 + tf.divide(self.other ,self.real)), lambda: tf.maximum(0.0, self.other-self.real+self.CONFIDENCE))

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))

        # prepare the list of all valid variables
        self.image_size = image_size
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.used_var_list = np.zeros(var_size, dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype = np.float32)
        self.modifier_down = np.zeros(var_size, dtype = np.float32)

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
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype = np.float32)
        self.vt = np.zeros(var_size, dtype = np.float32)
        self.adam_epoch = np.ones(var_size, dtype = np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

    def blackbox_optimizer_ordered_domain(self, iteration,ord_domain):
        # build new inputs, based on current variable value
        var = self.real_modifier.copy()
        var_size = self.real_modifier.size

        NN   = self.var_list.size
        nn   = self.batch_size

        if ((iteration+1)*nn<=NN):
            var_indice = ord_domain[iteration*nn : (iteration+1)*nn]
        else:
            var_indice = ord_domain[list(range(iteration*nn,NN)) + list(range(0,(self.batch_size-(NN-iteration*nn))))]  #check if this is true

        indice = self.var_list[var_indice]
        opt_fun = Objfun(lambda c: self.sess.run([self.loss],feed_dict = {self.use_log: self.use_log2,
                            self.modifier: vec2mod(c,indice,var)})[0])

        x_o = np.zeros(self.batch_size,)
        b   = self.modifier_up[indice]
        a   = self.modifier_down[indice]

        soln = pybobyqa.solve(opt_fun, x_o, rhobeg = self.L_inf/3,
                          bounds = (a,b),maxfun=self.q + 5,
                          npt=self.q+1)
        # print(soln)
        evaluations = soln.nf
        # adjust sample probability, sample around the points with large gradient
        nimgs = vec2mod(soln.x,indice,var)

        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)
        
        summary = opt_fun.get_summary(with_xs=False)

        distance = self.sess.run(self.distance, feed_dict={self.use_log : self.use_log2,self.modifier:nimgs})

        return soln.f, evaluations, nimgs, summary

    def blackbox_optimizer_GEN_ATTACK(self,iteration):
        # application of the gen attack according to the algorithm explained in
        # the article gen attack

        x_o = self.real_modifier.copy()
        x_o_shape = x_o.shape

        de  = self.L_inf
        rho = 5e-2

        G   = self.Generations
        N   = self.batch_size

        Pn  = []
        Pp  = []

        count = 0

        for i in range(N):
            temp  = x_o.copy()
            temp2 = temp.reshape(-1,)
            for j in range(len(temp2)):
                temp2[j] += 2*de*(np.random.rand()-0.5)*np.random.binomial(1,rho,1)

            Pn.append(temp2.reshape(temp.shape))

        for i in range(N-1):
            if np.all(Pn[i]==Pn[i+1]):
                print('Ugualgianza in',i)

        opt_fun = lambda c: self.sess.run([self.loss],feed_dict = {self.use_log: self.use_log2,self.modifier: c})

        output_fun = lambda c: self.sess.run([self.output],feed_dict = {self.use_log: self.use_log2,self.modifier: c})

        for g in range(G):
            F   = []
            out = []
            for i in range(N):
                temp_F = opt_fun(Pn[i].reshape(x_o_shape))
#                 print(temp_F)
                temp_out = output_fun(Pn[i].reshape(x_o_shape))
                F.append(-temp_F[0])
                out.append(temp_out[0])
                count += 1
            if int(count/6*5) > self.max_eval:
                return -4,count,Pn[0]
            if g%100 == 0:
                print('F maximum is: ',max(F),' with ', count/6*5, 'samples')
            #     print('The differnece is: ',max(F) - min(F))
            #     if g !=0:
            #         print('The spectrum is: ', probs)
            maxi = np.argmax(F)
            Pp = Pn

            if np.argmax(out[maxi][0]) == self.target :
                return 0, count, Pp[maxi]

            Pn = [Pp[maxi]]

            probs = np.divide(F - np.min(F),np.sum(F - np.min(F)))

            for i in range(0,N):
                n1      = np.random.choice(np.arange(1, N+1), p=probs.reshape(-1,))-1
                n2      = np.random.choice(np.arange(1, N+1), p=probs.reshape(-1,))-1

                parent1 = Pp[n1]
                parent2 = Pp[n2]
                F1      = probs[n1]#abs(F[n2])
                F2      = probs[n2]#abs(F[n1])

                child   = CrossOver(parent1,parent2,F1,F2).reshape(-1,)

                for j in range(len(child)):
                    child[j] += 2*de*(np.random.rand()-0.5)*np.random.binomial(1,rho,1)
                    if child[j]>self.L_inf:
                        child[j] = self.L_inf
                    if child[j]<-self.L_inf:
                        child[j] = -self.L_inf

                Pn.append(child.reshape(x_o_shape))
        print('We allegedely did ', count/6*5)
        if int(count/6*5) >= self.max_eval:
            return -4,count,Pn[0]
        return -3, count/6*5, Pn[0]

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
        if self.use_resize:
            self.resize_img(self.resize_init_size, self.resize_init_size, True)
        else:
            self.real_modifier.fill(0.0)

        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = img
        eval_costs = 0

    
        print(np.random.choice(10,3))
        ord_domain = np.random.choice(self.var_list.size, self.var_list.size, replace=False, p = self.sample_prob)

        started_with_log_now_normal = False

        self.sess.run(self.setup, {self.assign_timg: img,
                                   self.assign_tlab: lab})

        global_summary = []
        for step in range(self.MAX_ITERATIONS):

            # use the model left by last constant change
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0

            # print out the losses every 10%
            if step%(self.print_every) == 0:
                # print(self.use_log2)
                loss, output = self.sess.run((self.distance,self.output), feed_dict={self.use_log : self.use_log2,self.modifier: self.real_modifier})
                print("[STATS  ][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}".format(step, eval_costs, train_timer, self.real_modifier.shape, loss[0]))
                sys.stdout.flush()

            attack_begin_time = time.time()
            # perform the attack

            zz = np.zeros(img.shape)
            ub = 0.5*np.ones(img.shape)
            lb = -ub

            
            scaled_modifier = self.sess.run([self.scaled_modifier],feed_dict = {self.use_log: self.use_log2, self.modifier: self.real_modifier})[0]


            if step == 0:
                scaled_modifier = img

            self.modifier_up   = np.maximum( np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.L_inf,
                                                        ub.reshape(-1,) - img.reshape(-1,))
                                            ,zz.reshape(-1,))
            self.modifier_down = np.minimum( np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.L_inf,
                                                        - img.reshape(-1,) + lb.reshape(-1,) )
                                            ,zz.reshape(-1,))
            if step>0:
                self.sess.run(self.setup, {self.assign_timg: img,
                                           self.assign_tlab: lab})
                self.real_modifier.fill(0.0)

            if self.BOBYQA:
                iteration = np.mod(step,self.use_var_len//self.batch_size)
                if (iteration == 0):
                    # We have to repermute the pixels if they are modifie
                    prob = image_region_importance(img).reshape(-1,)
                    ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p = prob)
                    
                l, evaluations, nimg,summary = self.blackbox_optimizer_ordered_domain(iteration,ord_domain)

            elif self.GenAttack:
                l, evaluations, nimg = self.blackbox_optimizer_GEN_ATTACK(step)
                if l == -4:
                    print('The algorithm did not converge')
                    return nimg, evaluations, []
                summary = []
            # print('------> The output loss is ',l)
            global_summary.append(summary)

            self.real_modifier = nimg

            adv = self.sess.run([self.scaled_modifier],feed_dict = {self.use_log: self.use_log2, self.modifier: self.real_modifier})[0]

            temp = np.minimum(self.modifier_up.reshape(-1,),
                         adv.reshape(-1,))
            adv  = temp
            adv = np.maximum(self.modifier_down,
                         adv)
            adv = adv.reshape((self.image_size,self.image_size,self.num_channels))
            img = img + adv

            eval_costs += evaluations

            # Find the score output
            loss_test, score, real, other = self.sess.run((self.loss,self.output,self.real,self.other), feed_dict={self.use_log: self.use_log2,self.modifier: nimg})


            score = score[0]

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
                o_bestattack = nimg

            # If loss if 0 return the result
            if eval_costs > self.max_eval :
                print('The algorithm did not converge')
                return o_bestattack, eval_costs, global_summary

            if self.use_log2:
                if l < np.log(2):
                    print("Early Stopping becuase minimum reached")
                    return o_bestattack, eval_costs, global_summary
            else:
                if l<=0:
                    print("Early Stopping becuase minimum reached")
                    return o_bestattack, eval_costs, global_summary

            train_timer += time.time() - attack_begin_time

        # return the best solution found
        return o_bestattack, eval_costs
