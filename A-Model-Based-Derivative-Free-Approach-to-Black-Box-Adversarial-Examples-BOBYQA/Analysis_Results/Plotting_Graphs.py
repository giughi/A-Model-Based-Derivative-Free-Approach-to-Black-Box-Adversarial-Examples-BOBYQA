
# coding: utf-8

# In[1]:


import os
import sys
import tensorflow as tf
import numpy as np
import random
import time

import pickle

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain


def generating_cumulative(list_arrays, name_arrays, m, max_eval, refinement,BATCH,title, quants=[0.5]):
    # We suppose that every list has the same length
    # Refinement is the number of intercepts we want in the graph
    n = len(list_arrays)
    # m = len(list_arrays[0])
    print(m)
    r = np.array(range(refinement+1))*max_eval/refinement
    
    if n!=len(name_arrays):
        print('The names and list are not the same')
        return(-1)
    
    M = np.zeros((n,len(r)))
    
    for i in range(n):
        for j in range(len(r)):
            M[i,j] = np.sum(list_arrays[i]<r[j])
    M = M/m
    fig  = plt.figure()

    plt.plot(r,np.transpose(M))
    plt.legend(name_arrays)
    plt.xlabel('Function Evaluations')
    plt.ylabel('CDF')
    plt.axis([0, max_eval, 0 ,1])
    
    fig.savefig(title+str(BATCH)+'.jpg')
    
    return M


def generating_quantiles(median, median_all, mean, mean_all, methods, dataset, length, u_bound, L_inf_var,
                         prename, quant):
    directory = f(dataset)

    fig = plt.figure()
    plt.hold()
    for method in methods:
        plt.plot(L_inf_var, median[method])
    plt.legend(tuple(methods))
    plt.xlabel('$\varepsilon$')
    plt.ylabel('# queries')
    fig.savefig(prename + '_quant_' + quant + '.jpg')
    fig.savefig(prename + '_quant_' + quant + '.eps')

    fig = plt.figure()
    plt.hold()
    for method in methods:
        plt.plot(L_inf_var, mean[method])
    plt.legend(tuple(methods))
    plt.xlabel('$\varepsilon$')
    plt.ylabel('# queries')
    fig.savefig(prename + '_mean.jpg')
    fig.savefig(prename + '_mean.eps')

    fig = plt.figure()
    plt.hold()
    for method in methods:
        plt.plot(L_inf_var, median_all[method])
    plt.legend(tuple(methods))
    plt.xlabel('$\varepsilon$')
    plt.ylabel('# queries')
    fig.savefig(prename + '_all_quant_' + quant + '.jpg')
    fig.savefig(prename + '_all_quant_' + quant + '.eps')

    fig = plt.figure()
    plt.hold()
    for method in methods:
        plt.plot(L_inf_var, mean_all[method])
    plt.legend(tuple(methods))
    plt.xlabel('$\varepsilon$')
    plt.ylabel('# queries')
    fig.savefig(prename + '_all_mean.jpg')
    fig.savefig(prename + '_all_mean.eps')
    return


def plot_the_cdf(directory='./', prename='distilled_dist_L_inf_', L_inf_var=[0.1], dataset='mnist',
                 forname='.txt', methods=['gene', 'dist'], u_bound=3000, quant=0.5):

    mean = {}
    mean_all = {}
    median = {}
    median_all = {}
    
    list_results = {}
    list_succesf = {}
    list_all = {}
    
    for L_inf in L_inf_var:

        # generate the list of elements for each method and the list of the succesfful attacks
        for method in methods:
            with open(directory + prename + dataset + '_' + method + '_' + str(L_inf)+forname, "rb") as fp:
                list_results[method] = pickle.load(fp)

            list_succ = []
            
            # find the succesfull attacks
            for j in range(len(list_results[method])):
                if list_results[method][j][0] < 3000:
                    list_succ.append(list_results[method][j][0])
            list_succesf[method] = list_succ

        # generate the lists of the attaacks where all of the methods converged
        converged = []
        for j in range(len(list_results[methods[0]])):
            boolean = True
            # check that all the methods took less than what required
            for method in methods:
                if (list_results[method][j][0] >= u_bound) and boolean:
                    boolean = False
            # if they all converged, add element to the list of converged
            if boolean:
                converged.append(j)
        # now save the list as a dictionary
        for method in methods:
            list_all_temp = []
            for j in converged:
                list_all_temp.append(list_results[method][j][0])
            list_all[method] = list_all_temp

        # Now let's compute the different measures
        for method in methods:
            median[method].append(np.quantile(list_succesf[method], quant))
            mean[method].append(np.mean(list_succesf[method]))

            median_all[method].append(np.quantile(list_all[method], quant))
            mean_all[method].append(np.mean(list_all[method]))

        generating_cumulative(list_succesf, list_all, methods, dataset, len(list_succ), u_bound, L_inf_var,
                              prename + dataset, L_inf)

    generating_quantiles(list_succesf, list_all, methods, dataset, len(list_succ), u_bound, L_inf_var,
                         prename + '_' + dataset, L_inf_var, quant)

