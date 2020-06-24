
"""
BOBYQA Attack
This script generates a class BlackBox_BOBYQA which allows the generation of adversary examples
throught the algorithm presented in the "A Model Based Derivative Free Approach to Black Box 
Adversarial Examples: BOBYQA" manuscript.
Specifically, this script is used to generate attacks to the Inception-v3 net trained in the 
normal case and uploaded as a .pb file.

"""
from __future__ import print_function

import sys
# import os
import tensorflow as tf
import numpy as np
import time

import pybobyqa
import pandas as pd

# Initialisation Coefficients

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be


class Objfun(object):
    """
    Object to keep track of all the specific evaluations of the attack
    """
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

def vec2modMatRand(c, indice, var, RandMatr, depend, b, a, overshoot):
    """
    With this function we want to mantain the optiomisation domain
    centered.
    """
    temp = var.copy().reshape(-1, )
    n = len(indice)
    for i in range(n):
        indices = finding_indices(depend.reshape(-1, ), indice[i])
        if overshoot:
            temp[indices] += c[i]*np.max((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
        else:
            temp[indices] += c[i]*((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
    # we have to clip the values to the boundaries
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    temp = temp.reshape(var.shape)
    return temp


#########################################################
# Functions related to the optimal sampling of an image #
#########################################################


def find_neighbours(r, c, k, n, m, R):
    # This computes the neihgbours of a pixels (r,c,k) in an image R^(n,m,R)
    # Note: We never consider differnt layers of the RGB
    neighbours = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ((r+i) >= 0) and ((r+i) < n):
                if ((c+j) >= 0) and ((c+j) < m):
                    if not((i, j) == (0, 0)):
                        neighbours.append([0, r+i, c+j, k])
    return neighbours


def get_variation(img, neighbours):
    list_val = []
    for i in range(len(neighbours)):
        list_val.append(img[neighbours[i][1]][neighbours[i][2]][neighbours[i][3]])
    sum_var = np.std(list_val)
    return sum_var

    
def total_img_var(row, col, k, img):
    n, m, RGB = img.shape
    neighbours = find_neighbours(row, col, k, n, m, RGB)
    total_var = get_variation(img, neighbours)
    return total_var
    
    
def image_region_importance(img):
    # This function obtaines the image as an imput and it computes the importance of the 
    # different regions. 
    # 
    # Inputs:
    # - img: tensor of the image that we are considering
    # Outputs:
    # - probablity_matrix: matrix with the ecorresponding probabilities.
    
    n, m, k = img.shape
    probability_matrix = np.zeros((n, m, k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, img)

    # We have to give a probability also to all the element that have zero variance
    # this implies that we will add to the whole matrix the minimum nonzero value, divided
    # by 100
    if len(np.nonzero(probability_matrix)[0])!=0:
        probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/100
    else:
        probability_matrix += 1
    
    # Normalise the probability matrix
    probability_matrix = probability_matrix/np.sum(probability_matrix)
    
    return probability_matrix

#########################################################
# Functions to subdivide into sub regions the full pixels#
#########################################################

def associate_block(A, i, j, k, nn_i, nn_j, association):
    for ii in range(int(i), int(i + nn_i)):
        for jj in range(int(j), int(j + nn_j)):
            A[0, ii, jj, k] = int(association)
    return A

def matr_subregions_division(var, n, partition, k, Renew):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of beloging for each supervariable with only a block composition
    :param var: Image that we are perturbin. This will have shape (1,m,m,3)
    :param n: Dimension of the super grid that we are using (n,n,3)
    :param n_upl: Dimension of the leftdim
    :param k: number of times that each pixels is allowed to be assigned  (n,n,3)
    :return: The matrix with the supervariable tto which each pixel belongs
    """
    A = var.copy()
    # print('Initial Partition', partition)
    if partition is None:
        partition = []
        nn_up = np.ceil(299/n)
        for i in range(n):
            partition.append(i*nn_up)
        partition.append(298)
    elif Renew and n<290:
        partition_ = [0]
        for i in range(len(partition)-1):
            partition_.append(np.ceil((partition[i+1]+partition[i])/2))
            partition_.append(partition[i+1])
        partition = partition_

    # check that the number of intervals is n
    if len(partition)!=n+1:
        print('----- WARNING: the partition is not exact')

    association = 0
    for k in range(3):
        for i in range(n):
            xi = partition[i]
            di = partition[i+1]-partition[i]
            for j in range(n):
                xj = partition[j]
                dj = partition[j+1]-partition[j]
                A = associate_block(A, xi, xj, k, di, dj, association)
                association += 1
    return np.array([A]), partition


def finding_indices(dependency, index):
    # This returns a boolean matrix with the elements that are equal to index
    return dependency == index #, axis=len(dependency.shape)-1)


class BlackBox_BOBYQA:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, max_iterations=MAX_ITERATIONS,
                 print_every=100, use_resize=False, L_inf=0.15, init_size=5,
                 max_eval=1e5, done_eval_costs=0, max_eval_internal=0, 
                 perturbed_img=0, ord_domain=0, steps_done=-1,
                 iteration_scale=0, image0 = None, over='over',
                 permutation = None):
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
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.target = 0
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size
        
        self.L_inf = L_inf
        self.use_resize = use_resize
        self.max_eval = max_eval

        if max_eval_internal==0:
            self.max_eval_internal = max_eval
        else:
            self.max_eval_internal = max_eval_internal   
        self.done_eval_costs = done_eval_costs
        self.steps_done = steps_done
        self.perturbed_img = perturbed_img
        self.ord_domain = ord_domain
        self.iteration_scale = iteration_scale
        self.image0 = image0

        # each batch has a different modifier value (see below) to evaluate
        single_shape = (image_size, image_size, num_channels)
        small_single_shape = (self.small_x, self.small_y, num_channels)

        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size
        if over == 'over':
            self.overshoot=True
        elif over == 'linear':
            self.overshoot=False
        else:
            print('ERRROR, NOT RIGHT CLASSIFICATION TERM')
        self.l  =  0 # this is to keep track of the loss function and check that we can always improve it
        self.permutation = permutation
        # if self.use_resize:
        self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
        # scaled up image
        self.scaled_modifier = tf.image.resize_images(self.modifier, [image_size, image_size],
                                                        method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        # operator used for resizing image
        self.resize_size_x = tf.placeholder(tf.int32)
        self.resize_size_y = tf.placeholder(tf.int32)
        self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
        self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y],
                                                align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
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
        self.real = tf.reduce_sum(self.tlab*self.output, 1)
        
        # (1-self.tlab)*self.output gets all Z values for other classes
        # Because soft Z values are negative, it is possible that all Z values are less than 0
        # and we mistakenly select the real class as the max. So we minus 10000 for real class
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000), 1)
        self.use_max = False
        self.sum = tf.reduce_sum((1-self.tlab)*self.output, 1)
        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            # The loss is log(1 + other/real) if use log is true, max(other - real) otherwise
            self.loss = tf.log(tf.divide(self.sum + 1e-30, self.real+1e-30))
            # self.loss_max = tf.log(tf.divide(self.other +1e-30,self.real+1e-30))
            self.distance = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)

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
        
    def resize_img(self, small_x, small_y, reset_only=False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x,
                                                                          self.resize_size_y: self.small_y,
                                                                          self.resize_input: self.real_modifier})
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
                
    def blackbox_optimizer_ordered_domain(self, iteration, ord_domain, Random_Matrix, super_dependency, img, k, img0):
        """
        Function implementing the model-based minimisation over a batch of variables according to the ordering 
        ord_domain.
        """
        
        # build new inputs, based on current variable value
        times = np.zeros(8,)
        var = 0*np.array([img])
        # print('the type of var is', type(var[0]))
        # print('the shape of var is',var[0].shape)

        NN = self.var_list.size

        if len(ord_domain)<self.batch_size:
            nn = len(ord_domain)
        else:
            nn = self.batch_size

        # We choose the elements of ord_domain that are inherent to the step. So it is already
        # limited to the variable's dimension
        # print('inner iteration', iteration)
        if (iteration+1)*nn <= NN:
            var_indice = ord_domain[iteration*nn: (iteration+1)*nn]
        else:
            var_indice = ord_domain[list(range(iteration*nn, NN)) + 
                                    list(range(0, (self.batch_size-(NN-iteration*nn))))]
        
        indice = self.var_list[var_indice]
        x_o = np.zeros(nn,)
        # Changing the bounds according to the problem being resized or not
        if self.use_resize:
            a = np.zeros((nn,))
            b = np.zeros((nn,))

            for i in range(nn):
                indices = finding_indices(super_dependency.reshape(-1, ), indice[i])
                up = self.modifier_up[indices] 
                down = self.modifier_down[indices] 
                a[i] = -1
                b[i] = 1
                max_ind = np.argmax(up-down)
                xs =  np.divide( -(up+down),
                                (up-down))
                x_o[i] = np.clip(xs[max_ind],-1,1)
        
        else:
            b = self.modifier_up[indice]
            a = self.modifier_down[indice]
        bb = self.modifier_up
        aa = self.modifier_down
        opt_fun = Objfun(lambda c: self.sess.run([self.loss], feed_dict={
            self.modifier: vec2modMatRand(c, indice, var, Random_Matrix, super_dependency, bb, aa, self.overshoot)})[0])
        initial_loss = opt_fun(x_o)
        user_params = {'init.random_initial_directions':False,
                       'init.random_directions_make_orthogonal':False}
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg=np.min(b-a)/3,
                              bounds=(a, b), maxfun=nn*1.3,
                              rhoend=np.min(b-a)/6,
                              npt=nn+1, scaling_within_bounds=False,
                              user_params=user_params)
        summary = opt_fun.get_summary(with_xs=False)
        minimiser = np.min(summary['fvals'])
        evaluations = soln.nf
        
        nimgs = vec2modMatRand(soln.x, indice, var, Random_Matrix, super_dependency, bb, aa, self.overshoot)
        nimg2 = nimgs.copy()
        nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa] = bb[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa]
        nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa] = aa[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa]
        
        distance = self.sess.run(self.loss, feed_dict={self.modifier: nimgs})
        distance2 = self.sess.run(self.loss, feed_dict={self.modifier: nimg2})
        
        if soln.f > initial_loss:
            print('The optimisation is not working. THe diff is ', initial_loss - soln.f)
            return initial_loss, evaluations + 2, var, times, summary
        elif distance2 < distance:
            return distance2[0], evaluations + 2, nimg2, times, summary
        else:
            # print('The optimisation is working. THe diff is ', initial_loss - soln.f)
            return distance[0], evaluations + 2, nimgs, times, summary

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

        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)

        img0 = self.image0

        if self.use_resize:
            dimen = [2, 4, 8, 16, 32, 64, 128]
            steps = [0]
            for d in dimen[:-1]:
                n_var = d**2 * 3
                n_step = np.ceil(n_var/self.batch_size)
                steps.append(steps[-1] + n_step)

        else:
            steps = [0]
            dimen = [298]

        # clear the modifier
        self.resize_img(self.resize_init_size, self.resize_init_size, True)

        o_bestl2 = 1e10
        o_bestattack = img
        eval_costs = self.done_eval_costs
        internal_eval_costs = 0
        
        ord_domain = np.random.choice(self.var_list.size, self.var_list.size, replace=False, p=self.sample_prob)

        iteration_scale = -1
        iteration_domain = -1
        # print('There are at most ', self.MAX_ITERATIONS, ' iterations.')
        permutation=self.permutation
        if np.any(self.perturbed_img != 0):
            img = self.perturbed_img
            steps_inner = np.array(steps)
            idx = np.where(steps_inner == np.max(steps_inner[steps_inner<=self.steps_done+1]))[0][0]
            Random_Matrix = np.ones(np.array([img]).shape)
            self.small_x = dimen[idx]
            self.small_y = dimen[idx]
            self.resize_img(self.small_y, self.small_x, False)
            force_renew = False
            ord_domain = self.ord_domain
            iteration_scale = self.iteration_scale
            iteration_domain = np.mod(iteration_scale, (self.use_var_len//self.batch_size + 1))
            super_dependency, permutation = matr_subregions_division(np.zeros(np.array([img]).shape),
                                                                self.small_x, permutation, 1,False)

        
        self.sess.run(self.setup, {self.assign_timg: img, self.assign_tlab: lab})
        
        previous_loss = 1e6
        count_steady = 0
        global_summary = []
        adv = 0 * img
        values = {}
        for step in range(self.steps_done+1, self.MAX_ITERATIONS):
            # use the model left by last constant change
            train_timer = 0.0

            # print out the losses every print_every steps
            if step % self.print_every == 0:
                loss, output, distance = self.sess.run((self.loss, self.output, self.distance),
                                                       feed_dict={self.modifier: np.array([adv])})
                print("[STATS][L2] iter = {}, cost = {}, iter_sc = {:.3f}, iter_do = {:.3f}, size = {}, loss = {:.5g}, loss_f = {:.5g}, "
                      "maxel = {}".format(step, eval_costs, iteration_scale, iteration_domain, self.real_modifier.shape, distance[0], loss[0],
                                          np.argmax(output[0])))
                sys.stdout.flush()
                l = loss[0]
                self.l = l
            attack_begin_time = time.time()
            
            zz = np.zeros(img.shape)
            ub = 0.5*np.ones(img.shape)
            lb = -ub
            
            zz = np.zeros((self.image_size, self.image_size, self.num_channels))
            ub = 0.5*np.ones((self.image_size, self.image_size, self.num_channels))
            lb = -ub
            
            self.modifier_up = np.maximum(np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.L_inf,
                                                        ub.reshape(-1,) - img.reshape(-1,)),
                                            zz.reshape(-1,))
            self.modifier_down = np.minimum(np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.L_inf,
                                                        - img.reshape(-1,) + lb.reshape(-1,)),
                                            zz.reshape(-1,))
            
            if step > 0:
                self.sess.run(self.setup, {self.assign_timg: img,
                                           self.assign_tlab: lab})
                self.real_modifier.fill(0.0)
            

            iteration_scale += 1
            iteration_domain = np.mod(iteration_scale, (self.use_var_len//self.batch_size + 1))

            force_renew = False
            if iteration_domain == 0:
                # We force to regenerate a distribution of the nodes if we have 
                # already optimised over all of the domain
                force_renew = True

            if step in steps:
                idx = steps.index(step)
                self.small_x = dimen[idx]
                self.small_y = dimen[idx]
                self.resize_img(self.small_y, self.small_x, False)
                iteration_scale = 0
                iteration_domain = 0
                force_renew = True

            if  force_renew:  
                # We have to update the dependency matrix
                KK = 1
                super_dependency, permutation = matr_subregions_division(np.zeros(np.array([img]).shape),
                                                            self.small_x, permutation, KK, force_renew)
                Random_Matrix = np.ones(np.array([img]).shape)

                prob = image_region_importance(tf.image.resize_images(img, [self.small_x, self.small_y],
                                                                        align_corners=True,
                                                                        method=tf.image.ResizeMethod.BILINEAR)
                                                .eval()).reshape(-1,)
                print('Adding new ord_domain')
                ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p=prob)


            l, evaluations, nimg, times, summary = self.blackbox_optimizer_ordered_domain(iteration_domain,
                                                                                            ord_domain,
                                                                                            Random_Matrix,
                                                                                            super_dependency,
                                                                                            img, 1,img0)

            self.l = l

            global_summary.append(summary)
            
            adv = nimg[0]
            adv = adv.reshape((self.image_size, self.image_size, self.num_channels))
            img = img + adv
            eval_costs += evaluations
            internal_eval_costs += evaluations


            # Find the score output
            dist, l2, score, real, other = self.sess.run((self.distance, self.loss, self.output, self.real, self.other),
                                                          feed_dict={self.modifier: np.array([adv])})
                
            score = score[0]

            if dist < o_bestl2 and compare(score, np.argmax(lab)):
                # print a message if it is the first attack found
                if o_bestl2 == 1e10:
                    print("[STATS][L3](First valid attack found!) iter = {}, cost = {}".format(step, eval_costs))
                    sys.stdout.flush()
                o_bestl2 = l
                o_bestattack = img
            

            if internal_eval_costs > self.max_eval_internal:
                print('[STATS][L4] Rebuilding the net')
                values={'steps':step, 'iteration_scale':iteration_scale, 'ord_domain':ord_domain, 'loss':l, 'permutation':permutation}
                return img, eval_costs, global_summary, False, values

            # If loss if 0 return the result
            if eval_costs > self.max_eval:
                print('The algorithm did not converge')
                values={'steps':step, 'iteration_scale':iteration_scale, 'ord_domain':ord_domain, 'loss':l, 'permutation':permutation}
                return img, eval_costs, global_summary, True, values

            if dist <= 0:
                values={'steps':step, 'iteration_scale':iteration_scale, 'ord_domain':ord_domain, 'loss':l, 'permutation':permutation}
                print("Early Stopping becuase minimum reached")
                return o_bestattack, eval_costs, global_summary, True, values

            train_timer += time.time() - attack_begin_time

        return o_bestattack, eval_costs, global_summary, False, values
