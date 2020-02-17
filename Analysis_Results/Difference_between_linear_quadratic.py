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
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--max_iter", default=8000, help="Maximum iterations allowed", type=int)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

L_inf_var = 0.1
saving = args.save
maxiter = args.max_iter

loss_y = []
loss_n = []

    
with open(main_dir+'/Results/MNIST/summary_dist_L_inf_0.3_max_eval_1400_q_None.txt', "rb") as fp:
    L_y = pickle.load(fp)
with open(main_dir+'/Results/MNIST/summary_dist_L_inf_0.3_max_eval_1400_q_101.txt', "rb") as fp:
    L_n = pickle.load(fp)

# unwrapping the summaries

def unwrapping_MNIST(L):
    loss_y = []
    # print('=================',len(L))
    for iteration in range(len(L)):
        # print(itera,iteration)
        temp = L[iteration]['fvals'].values
        for j in range(len(temp)):
            loss_y.append(temp[j][0])

    return loss_y


loss_y = unwrapping_MNIST(L_y[0])
loss_n = unwrapping_MNIST(L_n[0])

x_y = np.arange(len(loss_y))
x_n = np.arange(len(loss_n))

saving_title=main_dir+'/Results/MNIST/Plots/linear_vs_quad.pdf'

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rc('text',usetex = True)
                
fig  = plt.figure()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# plt.plot(r,np.transpose(M[:3]),lw=2.5)
plt.plot(x_y,loss_y,lw=2.5,linestyle='-')
plt.plot(x_n,loss_n/loss_n[0]*loss_y[0],lw=2.5,linestyle='-')

plt.legend(['q=n+1','q=2n+1'], fontsize=18, framealpha=0.4, loc=1)
plt.xlabel("Queries",fontsize=18)
plt.ylabel(r"$\mathcal{L}$",fontsize=18)
plt.axis([0,np.minimum(len(loss_y), len(loss_n)),1.5,loss_y[0]],fontsize=18)

fig.savefig(saving_title,bbox_inches='tight')