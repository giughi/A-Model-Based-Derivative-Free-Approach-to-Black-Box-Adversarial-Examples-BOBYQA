from __future__ import print_function

import sys
# import os
import tensorflow as tf
import numpy as np
import time

import pybobyqa
import pandas as pd

import cv2
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


def vec2modMatRand3(c, indice, var, RandMatr, depend, b, a, overshoot):
    """
    With this function we want to mantain the optiomisation domain
    centered.
    """
    temp = var.copy().reshape(-1, )
    n = len(indice)
    for i in range(n):
        indices = finding_indices(depend.reshape(-1, ), indice[i])
        if overshoot:
            temp[indices] += np.tanh(c[i])*np.max((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
        else:
            temp[indices] += np.tanh(c[i])*((b-a).reshape(-1, )[indices])/2 + (b+a).reshape(-1, )[indices]/2
    # we have to clip the values to the boundaries
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    # temp[b-temp<temp-a] = b[b-temp<temp-a]
    # temp[b-temp>temp-a] = a[b-temp>temp-a]
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
    # print(type(img))
   
    probability_matrix = np.zeros((n, m, k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, img)
#                 print(probability_matrix[row,col,i])
    
    # We have to give a probability also to all the element that have zero variance
    # this implies that we will add to the whole matrix the minimum nonzero value, divided
    # by 100
    probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/100
    
    # Normalise the probability matrix
    probability_matrix = probability_matrix/np.sum(probability_matrix)
    
    return probability_matrix

#########################################################
# Functions to subdivide into sub regions the full pixels#
#########################################################


def nearest_of_the_list(i, j, k, n, m):
    """
    :param i: Row of the mxmx3 matrix in which we are
    :param j: Column of the mxmx3 Matrix in which we are
    :param k: Channel ...
    :param n: Dimension of the super-variable
    :param m: Dimension of the background matrix (n<m)
    :return: The relative elelemt of the super variable that is associated to i,j,k
    """
    x = np.linspace(0, 1, n)
    xx = np.linspace(0, 1, m)
    position_layer_x = np.argmin(np.abs(x-xx[j]))
    position_layer_y = np.argmin(np.abs(x-xx[i]))
    position_layer = position_layer_y*n + position_layer_x
    position_chann = k*n*n
    return position_layer + position_chann


def matr_subregions(var, n):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of beloging for each supervariable
    :param var: Image that we are perturbin. This will have shape (1,m,m,3)
    :param n: Dimension of the super grid that we are using
    :return: The matrix with the supervariable tto which each pixel belongs
    """
    A = var.copy()
    _, _, m, _ = A.shape
    for i in range(m):
        for j in range(m):
            for k in range(3):
                A[0, i, j, k] = nearest_of_the_list(i, j, k, n, m)
    return A


def associate_block(A, i, j, k, nn_i, nn_j, association):
    for ii in range(int(i), int(i + nn_i)):
        for jj in range(int(j), int(j + nn_j)):
            A[0, ii, jj, k] = int(association)
    return A

def matr_subregions_division(var, n, k, img_size):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of beloging for each supervariable with only a block composition
    :param var: Image that we are perturbin. This will have shape (1,m,m,3)
    :param n: Dimension of the super grid that we are using (n,n,3)
    :param k: number of times that each pixels is allowed to be assigned  (n,n,3)
    :return: The matrix with the supervariable tto which each pixel belongs
    """
    A = var.copy()

    # times the we will possibly 

    nn_up = np.floor(img_size/n)
    # We have to approximate it to the nearest 2 power
    if int(nn_up) in [74,37]:
        if nn_up == 74:
            nn_up=72
        else:
            nn_up=36
        # print('nn_up ', nn_up)


    nn_do = img_size - nn_up*(n-1) 
    if nn_do <=0:
        nn_up = nn_up-1
        nn_do = img_size - nn_up*(n-1)
    
    association = 0
    for k in range(3):
        nn_i = nn_up
        for i in range(n):
            if i == n-1:
                nn_i = nn_do
            nn_j = nn_up
            for j in range(n):
                if j == n - 1:
                    nn_j = nn_do
                A = associate_block(A, i*nn_up, j*nn_up, k, nn_i, nn_j, association)
                association += 1

    return np.array([A])

def matr_subregions_division_2(var, n, partition, k, Renew,img_size):
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
        nn_up = np.ceil(img_size/n)
        for i in range(n):
            partition.append(i*nn_up)
        partition.append(img_size)

    elif Renew and ((n<290 and img_size==299) or img_size<200):
        partition_ = [0]
        for i in range(len(partition)-1):
            partition_.append(np.ceil((partition[i+1]+partition[i])/2))
            partition_.append(partition[i+1])
        partition = partition_

    if n>=img_size:
        partition = []
        nn_up = 1
        for i in range(n):
            partition.append(i*nn_up)
        partition.append(img_size)
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
    print(partition)
    return np.array([A]), partition


def finding_indices(dependency, index):
    # This returns a boolean matrix with the elements that are equal to index
    return dependency == index #, axis=len(dependency.shape)-1)


class BlackBox_BOBYQA:
    def __init__(self, loss_f, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, max_iterations=MAX_ITERATIONS,
                 print_every=100, early_stop_iters=0,
                 use_resize=False, L_inf=0.15, init_size=5,
                 max_eval=1e5, done_eval_costs=0, max_eval_internal=0, 
                 perturbed_img=0, ord_domain=0, steps_done=-1,
                 iteration_scale=0, image0 = None, over='over',
                 permutation = None, image_size=299, num_channels=3, num_labels=1001):
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
        self.loss_f = loss_f
        self.TARGETED = targeted
        self.target = 0
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        if use_resize:
            self.small_x = self.resize_init_size-1
            self.small_y = self.resize_init_size-1
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
        # the real variable, initialized to 0
        self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        
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
            # print(self.real_modifier.shape, small_x, self.real_modifier)
            self.real_modifier = np.array([cv2.resize(self.real_modifier[0], (small_x, small_y))])
            # print(self.real_modifier.shape)
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
                
    def blackbox_optimizer_ordered_domain(self, iteration, ord_domain, Random_Matrix, super_dependency, img, k, img0):
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
        
        # print('======> optimised indices', var_indice)
        indice = self.var_list[var_indice]
        x_o = np.zeros(nn,)
        # Changing the bounds according to the problem being resized or not
        if self.use_resize:
            a = -2*np.ones((nn,))
            b = +2*np.ones((nn,))

            for i in range(nn):
                indices = finding_indices(super_dependency.reshape(-1, ), indice[i])
                up = self.modifier_up[indices] 
                down = self.modifier_down[indices] 
                # a[i] = -1
                # b[i] = 1
                max_ind = np.argmax(up-down)
                xs =  np.divide( -(up+down),
                                (up-down))
                x_o[i] = np.arctanh(np.clip(xs[max_ind],-1,1))
                # print(x_o[i])
        
        else:
            b = self.modifier_up[indice]
            a = self.modifier_down[indice]
        bb = self.modifier_up
        aa = self.modifier_down
        opt_fun = Objfun(lambda c: self.loss_f(img, vec2modMatRand3(c, indice, var, Random_Matrix, super_dependency, bb, aa,
                                                               self.overshoot), only_loss=True)[0])
        initial_loss = opt_fun(x_o)
        # if initial_loss != self.l:
        #     print(' COULD NOT REBUILD pert', initial_loss-self.l)
        user_params = {'init.random_initial_directions':False, 'init.random_directions_make_orthogonal':False}
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg=np.min(b-a)/3,
                              bounds=(a, b), maxfun=nn*1.2,
                              rhoend=np.min(b-a)/6,
                              npt=nn+1, scaling_within_bounds=False,
                              user_params=user_params)
        summary = opt_fun.get_summary(with_xs=True)
        minimiser = np.min(summary['fvals'])
        real_oe = self.loss_f(img, vec2modMatRand3(soln.x, indice, var, Random_Matrix, super_dependency, bb, aa,
                                                               self.overshoot), only_loss=True)
        if (minimiser != real_oe) and (initial_loss>minimiser):
            print('########################## ERRRORROR')
        # print(a,b)
        evaluations = soln.nf
        
        # print('==========   a  ', a)
        # print('==========   b  ', b)
        # print('========== soln ', soln.x)
        # print(soln)

        nimgs = vec2modMatRand3(soln.x, indice, var, Random_Matrix, super_dependency, bb, aa, self.overshoot)
        nimg2 = nimgs.copy()
        nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa] = bb[bb-nimgs.reshape(-1,)<nimgs.reshape(-1,)-aa]
        nimg2.reshape(-1,)[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa] = aa[bb-nimgs.reshape(-1,)>nimgs.reshape(-1,)-aa]
        
        
        distance = self.loss_f(img,nimgs, only_loss=True)
        distance2 = self.loss_f(img, nimg2, only_loss=True)

        if soln.f > initial_loss:
            print('The optimisation is not working. THe diff is ', initial_loss - soln.f)
            return initial_loss, evaluations + 2, var, times, summary
        elif distance2 < distance:
            print('USING ROUNDED')
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
            # steps = [0, 1, 5, 21, 83, 321]
            dimen = [2, 4, 8, 16, 32, 64, 128]
            steps = [0]
            for d in dimen[:-1]:
                n_var = d**2 * 3
                n_step = np.ceil(n_var/self.batch_size)
                steps.append(steps[-1] + n_step)

            # print('Dimen', dimen)
            # print('Steps', steps)
        else:
            steps = [0]
            dimen = [298]
        # n_var = [48, 192, 768, ]

        # clear the modifier
        # if self.use_resize:
        self.resize_init_size = dimen[0]
        self.resize_img(self.resize_init_size, self.resize_init_size, True)
        # else:
        #     self.real_modifier.fill(0.0)


        # the best l2, score, and image attack
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
            super_dependency, permutation = matr_subregions_division_2(np.zeros(np.array([img]).shape),
                                                                self.small_x, permutation, 1,False, self.image_size)

        global_summary = []
        adv = 0 * img
        values = {}
        for step in range(self.steps_done+1, self.MAX_ITERATIONS):
            # use the model left by last constant change
            train_timer = 0.0

            # print out the losses every 10%
            # print('-----> Adv is of type', type(adv), ' and shape', adv.shape)
            if step % self.print_every == 0:
                loss, output, distance= self.loss_f(img, np.array([adv]))
                # print(loss, output, distance)
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
            
            # if self.use_resize:
            zz = np.zeros((self.image_size, self.image_size, self.num_channels))
            ub = 0.5*np.ones((self.image_size, self.image_size, self.num_channels))
            lb = -ub
            
            self.modifier_up = np.maximum(np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.L_inf,
                                                        ub.reshape(-1,) - img.reshape(-1,)),
                                            zz.reshape(-1,))
            self.modifier_down = np.minimum(np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.L_inf,
                                                        - img.reshape(-1,) + lb.reshape(-1,)),
                                            zz.reshape(-1,))
            
            # if step > 0:
            #     self.real_modifier.fill(0.0)
            
            iteration_scale += 1
            iteration_domain = np.mod(iteration_scale, (self.use_var_len//self.batch_size + 1))

            # print(iteration_domain,iteration_scale)
            force_renew = False
            # print('iteration_domain ', iteration_domain)
            if iteration_domain == 0:
                # We force to regenerate a distribution of the nodes if we have 
                # already optimised over all of the domain
                force_renew = True

            # if self.use_resize:
            if step in steps:
                idx = steps.index(step)
                self.small_x = dimen[idx]
                self.small_y = dimen[idx]
                self.resize_img(self.small_y, self.small_x, False)
                # print('Insider of reside dim', self.small_x)
                iteration_scale = 0
                iteration_domain = 0
                force_renew = True

            if  force_renew:  
                # We have to restrict the random matrix and the
                KK = 1
                super_dependency, permutation = matr_subregions_division_2(np.zeros(np.array([img]).shape),
                                                            self.small_x, permutation, KK, force_renew, self.image_size)

                Random_Matrix = np.ones(np.array([img]).shape)


                # if self.use_resize:
                prob = image_region_importance(cv2.resize(img, (self.small_x, self.small_y), 
                                                            interpolation=cv2.INTER_LINEAR)).reshape(-1,)

                print('Adding new ord_domain')
                ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p=prob)
                

            # print('====> ord_domain', ord_domain)
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

            adv= 0*adv

            loss, output, distance = self.loss_f(img, np.array([adv]))
            score = output[0]

            if distance[0] < o_bestl2 and compare(score, np.argmax(lab)):
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

            if distance[0] <= 0:
                values={'steps':step, 'iteration_scale':iteration_scale, 'ord_domain':ord_domain, 'loss':l, 'permutation':permutation}
                print("Early Stopping becuase minimum reached")
                return o_bestattack, eval_costs, global_summary, True, values

            train_timer += time.time() - attack_begin_time

        return o_bestattack, eval_costs, global_summary, False, values
