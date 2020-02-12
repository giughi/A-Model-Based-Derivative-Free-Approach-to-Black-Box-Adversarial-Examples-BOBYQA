"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import time
import random
import numpy as np
import tensorflow as tf 
import cv2
# from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class GenAttack2(object):
    def mutation_op(self,  cur_pop, idx, step_noise=0.01, p=0.005):
        perturb_noise = np.random.random_sample(cur_pop.shape)*(2*step_noise)-step_noise
        mutated_pop = cur_pop
        indices = np.random.random_sample(cur_pop.shape) < p
        mutated_pop[indices]  += perturb_noise[indices] 
        return mutated_pop


    def attack_step(self, idx, success, orig_copies, cur_noise, best_win_margin, cur_plateau_count, num_plateaus):
        if self.resize_dim:
            noise_resized = []
            for i in range(len(cur_noise)):
                noise_resized.append(cv2.resize(cur_noise[i], (299, 299)))
            noise_resized = np.array(noise_resized)
        else:
            noise_resized = cur_noise
        noise_dim = self.resize_dim or 299
        cur_pop = np.clip(noise_resized + orig_copies, self.box_min, self.box_max)
        time_befor = time.time()
        pop_preds, loss, difference = self.loss_f(cur_pop)  # these are the probs, loss, difference
        time_after = time.time()
        all_preds = np.argmax(pop_preds, axis=1)

        success_pop = (all_preds == self.target)
        success = np.max(success_pop, axis=0)

        win_margin = np.max(difference)
        
        if win_margin > best_win_margin:
            new_best_win_margin, new_cur_plateau_count = (win_margin, 0)
        else:
            new_best_win_margin, new_cur_plateau_count = (best_win_margin, cur_plateau_count+1)
                    
        if win_margin>-0.4:           
            plateau_threshold = 100
        else:
            plateau_threshold = 300

        if new_cur_plateau_count > plateau_threshold:
            new_num_plateaus, new_cur_plateau_count = (num_plateaus+1, 0)
        else:
            new_num_plateaus, new_cur_plateau_count = (num_plateaus, new_cur_plateau_count)

        if self.adaptive:
            step_noise =   np.maximum(self.alpha, 
                    0.4*np.power(0.9, new_num_plateaus))
            if idx < 10:
                step_p =  1.0
            else:
                step_p = np.maximum(self.mutation_rate, 0.5*np.power(0.90, new_num_plateaus))
        else:
            step_noise = self.alpha
            step_p = self.mutation_rate

        step_temp = 0.1


        if success ==1:
            elite_idx = np.expand_dims(np.argmax(success_pop), axis=0)
        else:
            elite_idx = np.expand_dims(np.argmax(loss, axis=0), axis=0)

        elite = cur_noise[elite_idx]
        select_probs = softmax(np.array(loss)/ step_temp)
        # print(loss,select_probs)
        parents = np.random.choice(self.pop_size, 2*self.pop_size-2, p=select_probs)
        # print('###########################', parents)
        # tf.contrib.distributions.Categorical(probs=select_probs).sample(2*self.pop_size-2)
        parent1 = cur_noise[parents[:self.pop_size-1]]
        parent2 = cur_noise[parents[self.pop_size-1:]]
        pp1 = select_probs[parents[:self.pop_size-1]]
        pp2 = select_probs[parents[self.pop_size-1:]]
        pp2 = pp2 / (pp1+pp2)
        # temp = []
        # for j in range(self.pop_size-1):
        #     temp.append(pp2[j]*np.ones((1, noise_dim, noise_dim,3)))
        # pp2 = np.array(temp)   #np.tile(pp2, (1, noise_dim, noise_dim,3))#pp2*np.ones((5,noise_dim,noise_dim,3)) #
        # print(' ========================= ', parent1.shape)
        prob = np.random.random_sample(pp2.shape)
        xover_prop = prob > pp2
        childs = []
        for j in range(len(xover_prop)):
            if xover_prop[j]:
                childs.append(parent1[j])
            else:
                childs.append(parent2[j])
        childs =  np.array(childs)
        
        idx +=1
        print([idx, np.min(loss), win_margin, step_p, step_noise, new_cur_plateau_count,time_after-time_befor])
        # print(childs.shape)
        mutated_childs = self.mutation_op(childs, idx=idx, step_noise=self.eps*step_noise, p=step_p)
        # print(elite.shape, mutated_childs.shape)
        new_pop = np.concatenate((mutated_childs, elite), axis=0)
        return idx, success, orig_copies, new_pop, new_best_win_margin, new_cur_plateau_count, new_num_plateaus, np.reshape(elite,(noise_dim, noise_dim, 3))


    def __init__(self, loss_f, pop_size=6, mutation_rate=0.001,
            eps=0.15, max_steps=10000, alpha=0.20,
            resize_dim=None, adaptive=False, num_classes=1001):
        self.eps = eps
        self.pop_size = pop_size
        self.alpha = alpha
        self.max_steps = max_steps
        self.mutation_rate = mutation_rate
        self.resize_dim = resize_dim
        self.noise_dim = self.resize_dim or 299
        self.adaptive = adaptive
        
        self.loss_f= loss_f


    def attack(self, input_img, target_label): 
        self.input_img = input_img
        self.target = target_label
        self.box_min = np.maximum(self.input_img-self.eps, -0.5)
        self.box_max = np.minimum(self.input_img+self.eps, 0.5)
        #self.initialize(sess, input_img, target_label) initialise the following [
        pop_orig = []
        for j in range(self.pop_size):
            pop_orig.append(input_img)
        pop_orig= np.array(pop_orig)
        print('========', pop_orig.shape)
        pop_noise = np.zeros((self.pop_size, self.noise_dim, self.noise_dim, 3))
        best_win_margin = -1
        cur_plateau_count = 0,
        num_plateaus = 0 
        i = 0
        success = 0
        while (i < self.max_steps) and not (success):
            i, success, pop_orig, pop_noise, best_win_margin, cur_plateau_count, num_plateaus, adv_noise = self.attack_step(
                                                        i, success, pop_orig, pop_noise, 
                                                        best_win_margin, cur_plateau_count, num_plateaus)
        num_steps=i


        if success:
            if self.resize_dim:
                adv_img = np.clip( np.array([input_img])+np.array([cv2.resize(adv_noise, (299,299))]),
                                  self.box_min, self.box_max)
            else:
                adv_img = np.clip(np.array([input_img])+np.expand_dims(adv_noise, axis=0),
                                  self.box_min, self.box_max)

            # Number of queries = NUM_STEPS * (POP_SIZE -1 ) + 1
            # We subtract 1 from pop_size, because we use elite mechanism, so one population 
            # member is copied from previous generation and no need to re-evaluate it.
            # The first population is an exception, therefore we add 1 to have total sum.
            query_count = num_steps * (self.pop_size  - 1)+ 1
            return success, adv_img[0], query_count
        else:
            query_count = num_steps * (self.pop_size  - 1)+ 1
            return success, 3, query_count
