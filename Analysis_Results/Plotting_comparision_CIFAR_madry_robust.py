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
# parser.add_argument("eps", default=0.1, help="Energy of the pertubation", type=float)
parser.add_argument("--eps", default=[0.1], nargs='+', type=float, help="Energy of the pertubation")
parser.add_argument("--Data", default='Imagenet', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR','STL10','Imagenet','MiniImageNet'")
parser.add_argument("--title", default='_madry', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--Adversary", default=False, help="Boolean for plotting adversarial attacks too", type=bool)
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=3000, help="Maximum iterations allowed", type=int)
parser.add_argument("--second_iter",default=False, help="Loading results from second round of attacks", type=bool)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

dataset = 'CIFAR'
L_inf_var = [0.05, 0.1,0.15,0.2]
adversary = args.Adversary
saving = args.save
plot_type = args.plot_type
quant = args.quantile
maxiter = args.max_iter
title = args.title

# distance    = []

SR_g = []
SR_c = []
SR_b = []
SR_s = []
SR_g_adv = []
SR_c_adv = []
SR_b_adv = []
SR_s_adv = []

for i in range(len(L_inf_var)):
    dist_tot = []
    gene_tot = []
    combi_tot = []
    block_tot = []

    dist_tot_adv = []
    gene_tot_adv = []
    combi_tot_adv = []
    block_tot_adv = []

    BATCH = L_inf_var[i]
    

    # with open(main_dir+'/Results/CIFAR/boby_L_inf_'+str(BATCH)+'_max_eval_300_madry.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    #     dist_adv = pickle.load(fp)
    # # with open(main_dir+'/Results/CIFAR/gene_L_inf_'+str(BATCH)+'_max_eval_300_madry.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    # #     gene_adv = pickle.load(fp)
    # with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_300_madry.txt',"rb") as fp:
    #     combi_adv = pickle.load(fp)
    # with open(main_dir+'/Results/CIFAR/square__L_inf_'+str(BATCH)+'_max_eval_300_madry.txt',"rb") as fp:
    #     block_adv = pickle.load(fp)
        
    

    with open(main_dir+'/Results/CIFAR/boby_L_inf_'+str(BATCH)+'_max_eval_3000_madry.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        dist = pickle.load(fp)
    # with open(main_dir+'/Results/CIFAR/gene_L_inf_'+str(BATCH)+'_max_eval_300_distilled.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    #     gene = pickle.load(fp)
    with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_3000_madri.txt',"rb") as fp:
        combi = pickle.load(fp)
    with open(main_dir+'/Results/CIFAR/square_NO_RAND_INIT_L_inf_'+str(BATCH)+'_max_eval_3000_madry.txt',"rb") as fp:
        block = pickle.load(fp)

    total_number = np.minimum(np.minimum(len(dist), len(block)), len(combi))
    dist = dist[:total_number]
    combi = combi[:total_number]
    # gene = gene[:total_number]
    block = block[:total_number]

    list_dist_2 = []
    # list_gene_2 = []
    list_combi_2 = []
    list_block_2 = []
        
    for j in range(np.minimum(len(dist), 
                              np.minimum(len(combi),len(block))
                             )):
        dist_tot.append(dist[j][0])
        # gene_tot.append(gene[j][0])
        combi_tot.append(combi[j][0])
        block_tot.append(block[j][0])    
        
        if dist[j][0]< maxiter:
            list_dist_2.append(dist[j][0])
        # if gene[j][0]< maxiter and gene[j][0] != 3000:
        #     list_gene_2.append(gene[j][0])
        if combi[j][0] < maxiter:
            list_combi_2.append(combi[j][0])
        if block[j][0] < maxiter:
            list_block_2.append(block[j][0])
    
    SR_b.append(len(list_dist_2)/len(dist_tot))
    # SR_g.append(len(list_gene_2)/len(gene_tot))
    SR_c.append(len(list_combi_2)/len(combi_tot))
    SR_s.append(len(list_block_2)/len(block_tot))
    print('Lengths ', str(BATCH), ': boby:{}, combi:{}, square:{}'.format(len(dist_tot), len(combi_tot), len(block_tot)))

print('Lengths: linear:{}, combi:{}, over:{}'.format(len(dist), len(combi), len(block)))

    


# Final plotting function

# list_gene = []
list_block = []
list_dist = []
list_combi = []


total_number = np.minimum(
                      np.minimum(
                             len(block),len(dist)
                             ),
                      np.minimum(
                             len(combi), len(combi)
                             )
                      )

for j in range(total_number):
    list_block.append(block[j][0])
    # list_gene.append(gene[j][0])
    list_dist.append(dist[j][0])
    list_combi.append(combi[j][0])

# if len(block)<total_number:

saving_title=main_dir+'/Results/'+dataset+'/Plots/Energy_performance'+title

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rc('text',usetex = True)
                
fig  = plt.figure()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

    
# print('Gen', SR_g)
print('COM', SR_c)
print('BOB', SR_b)
print('SQA', SR_s)
# print('Adversarial Attacks')
# # print('Gen', SR_g_adv)
# print('COM', SR_c_adv)
# print('BOB', SR_b_adv)
# print('SQA', SR_s_adv)


list_arrays = [list_dist_2,list_combi_2,
            #    list_gene_2_adv,list_dist_2_adv,list_combi_2_adv,
               list_block_2]

n = 4

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

name_arrays = ['BOBYQA','COMBI','SQUARE']

fig  = plt.figure()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# pl0, = plt.semilogx(L_inf_var,SR_g,lw=2,linestyle='-', marker='o',color=colors[0])
pl1, = plt.semilogx(L_inf_var,SR_b,lw=2,linestyle='--', marker='o',color=colors[1])
pl2, = plt.semilogx(L_inf_var,SR_c,lw=2,linestyle='-', marker='o',color=colors[2])
pl3, = plt.semilogx(L_inf_var,SR_s,lw=2,linestyle='-.', marker='o',color=colors[3])

# pl4, = plt.semilogx(L_inf_var,SR_g_adv,lw=2,linestyle='--', marker='x',color=colors[0])
# pl5, = plt.semilogx(L_inf_var,SR_b_adv,lw=2,linestyle='--', marker='x',color=colors[1])
# pl6, = plt.semilogx(L_inf_var,SR_c_adv,lw=2,linestyle='--', marker='x',color=colors[2])
# pl7, = plt.semilogx(L_inf_var,SR_s_adv,lw=2,linestyle='--', marker='x',color=colors[3])
l1 = plt.legend([pl1,pl2,pl3], name_arrays,loc=0, fontsize=16,ncol=1)
# l2 = plt.legend([pl4,pl5,pl6,pl7], name_arrays,loc=4, fontsize=16, framealpha=0.,ncol=1)

plt.gca().add_artist(l1)

plt.xlabel(r"$\epsilon_\infty$",fontsize=18)
plt.ylabel('SR',fontsize=18)
# plt.axis([0, max_eval, 0 ,1],fontsize=18)

# if LEGEND:
# plt.text(0.058,0.4,'Adv', fontsize=16)
# plt.text(0.047,0.4,'Norm', fontsize=16)

fig.savefig(saving_title+'.pdf',bbox_inches='tight')
