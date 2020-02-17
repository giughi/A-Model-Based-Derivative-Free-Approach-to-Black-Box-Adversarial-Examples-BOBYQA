import os
import sys
import numpy as np
import random
import time

import pickle

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
from itertools import chain


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", default=1400, help="Maximum iterations allowed", type=int)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

L_inf_var = 0.1
maxiter = args.max_iter

loss_y = []
loss_n = []

# Working on Imagenet

with open(main_dir+'/Results/Imagenet/Iterative_0.1_batch_dim_25_max_queries_1400_hier_True_2.txt', "rb") as fp:
    I_25 = pickle.load(fp)
with open(main_dir+'/Results/Imagenet/Iterative_0.1_batch_dim_50_max_queries_1400_hier_True_2.txt', "rb") as fp:
    I_50 = pickle.load(fp)
with open(main_dir+'/Results/Imagenet/Iterative_0.1_batch_dim_100_max_queries_1400_hier_True_2.txt', "rb") as fp:
    I_100 = pickle.load(fp)

print(len(I_50))

def unwrapping_ImageNet(L):
    iterations = len(L)
    losses = []
    lengths = []
    for itera in range(iterations):
        
        loss_y = []
        for i in range(len(L[itera])):
            for iteration in range(len(L[itera][i])):
                temp = L[itera][i][iteration]['fvals'].values
                for j in range(len(temp)):
                    loss_y.append(temp[j][0])
        losses.append(loss_y)
        
        len_y = len(loss_y)
        lengths.append(len_y)

    max_len = max(lengths)
    average_loss = []
    for j in range(max_len):
        temp = 0 
        count = 0
        for k in range(iterations):
            if j<lengths[k]:
                temp += losses[k][j]
            else:
                temp += losses[k][-1]
        average_loss.append(temp/iterations)
        
    return average_loss

loss_I_25 = unwrapping_ImageNet(I_25)
loss_I_50 = unwrapping_ImageNet(I_50)
loss_I_100 = unwrapping_ImageNet(I_100)

# Working on cifar

with open(main_dir+'/Results/CIFAR/summary_dist_L_inf_0.1_max_eval_1400_batch_25.txt', "rb") as fp:
    C_25 = pickle.load(fp)
with open(main_dir+'/Results/CIFAR/summary_dist_L_inf_0.1_max_eval_1400_batch_50.txt', "rb") as fp:
    C_50 = pickle.load(fp)
with open(main_dir+'/Results/CIFAR/summary_dist_L_inf_0.1_max_eval_1400_batch_100.txt', "rb") as fp:
    C_100 = pickle.load(fp)

def unwrapping_CIFAR(L):
    iterations = len(L)
    losses = []
    lengths = []
    for itera in range(iterations):
        loss_y = []
        for iteration in range(len(L[itera])):
            # print(itera,iteration)
            temp = L[itera][iteration]['fvals'].values
            for j in range(len(temp)):
                loss_y.append(temp[j][0])
        losses.append(loss_y)
        len_y = len(loss_y)
        lengths.append(len_y)

    max_len = max(lengths)
    average_loss = []
    for j in range(max_len):
        temp = 0 
        count = 0
        for k in range(iterations):
            if j<lengths[k]:
                temp += losses[k][j]
        average_loss.append(temp/iterations)
        
    return average_loss

loss_C_25 = unwrapping_CIFAR(C_25)
loss_C_50 = unwrapping_CIFAR(C_50)
# print(C_100)
loss_C_100 = unwrapping_CIFAR(C_100)

# unwrappign MNIST

with open(main_dir+'/Results/MNIST/summary_dist_L_inf_0.3_max_eval_1400_batch_25.txt', "rb") as fp:
    M_25 = pickle.load(fp)
with open(main_dir+'/Results/MNIST/summary_dist_L_inf_0.3_max_eval_1400_batch_50.txt', "rb") as fp:
    M_50 = pickle.load(fp)
with open(main_dir+'/Results/MNIST/summary_dist_L_inf_0.3_max_eval_1400_batch_100.txt', "rb") as fp:
    M_100 = pickle.load(fp)

loss_M_25 = unwrapping_CIFAR(M_25)
loss_M_50 = unwrapping_CIFAR(M_50)
loss_M_100 = unwrapping_CIFAR(M_100)

def plot_the_batches(loss_25, loss_50, loss_100, saving_name, legend):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.rc('text',usetex = True)
                    
    fig  = plt.figure()

    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)


    x_25 = np.arange(len(loss_25))
    x_50 = np.arange(len(loss_50))
    x_100 = np.arange(len(loss_100))

    min_x = 1350
    # plt.plot(r,np.transpose(M[:3]),lw=2.5)
    plt.plot(x_25,loss_25,lw=2.5,linestyle='-')
    plt.plot(x_50,loss_50,lw=2.5,linestyle='-')
    plt.plot(x_100,loss_100,lw=2.5,linestyle='-')

    minimal_loss = np.minimum(np.minimum(loss_25[min_x], loss_50[min_x]), loss_100[min_x])
    if legend:
        plt.legend(['b=25','b=50', 'b=100'], fontsize=22, framealpha=0.4)
    plt.xlabel("Queries",fontsize=22)
    plt.ylabel(r"$\mathcal{L}$",fontsize=22)
    plt.axis([0,min_x,minimal_loss,loss_25[0]],fontsize=22)

    fig.savefig(saving_name,bbox_inches='tight')

saving_title=main_dir+'/Results/Imagenet/Plots/Batch_Comparision.pdf'
plot_the_batches(loss_I_25, loss_I_50, loss_I_100, saving_title, False)

saving_title=main_dir+'/Results/CIFAR/Plots/Batch_Comparision.pdf'
plot_the_batches(loss_C_25, loss_C_50, loss_C_100, saving_title, True)

saving_title=main_dir+'/Results/MNIST/Plots/Batch_Comparision.pdf'
plot_the_batches(loss_M_25, loss_M_50, loss_M_100, saving_title, True)