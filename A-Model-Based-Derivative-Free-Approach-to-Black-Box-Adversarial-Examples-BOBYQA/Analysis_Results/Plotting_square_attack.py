import os
import sys
import numpy as np
import random
import time

import pickle

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain


import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("eps", default=0.1, help="Energy of the pertubation", type=float)
parser.add_argument("--eps", default=0.1, type=float, help="Energy of the pertubation")
parser.add_argument("--Data", default='Imagenet', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR','STL10','Imagenet','MiniImageNet'")
parser.add_argument("--title", default='_SQUARE', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--Adversary", default=False, help="Boolean for plotting adversarial attacks too", type=bool)
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=3000, help="Maximum iterations allowed", type=int)
parser.add_argument("--second_iter",default=True, help="Loading results from second round of attacks", type=bool)

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

dataset = args.Data 
L_inf_var = args.eps
adversary = args.Adversary
saving = args.save
plot_type = args.plot_type
quant = args.quantile
maxiter = args.max_iter
title = args.title
second_iter = args.second_iter

# distance    = []

dist_tot = []
gene_tot = []
combi_tot = []

dist_tot = []
gene_tot = []
combi_tot = []
BATCH = L_inf_var

if args.Data == 'cifar':
    with open(main_dir+'/Results/CIFAR/dist_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        dist = pickle.load(fp)
    with open(main_dir+'/Results/CIFAR/gene_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        gene = pickle.load(fp)
    with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_3000_normal.txt',"rb") as fp:
        combi = pickle.load(fp)
    with open(main_dir+'/Results/CIFAR/Square_summary_dist_L_inf_'+str(BATCH)+'.txt',"rb") as fp:
        block = pickle.load(fp)
else:
    BATCH = 0.05
    with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        dist = pickle.load(fp)
    with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        gene = pickle.load(fp)
    with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        combi = pickle.load(fp)
    with open(main_dir+'/Results/Imagenet/SQUA_'+str(BATCH)+'_targeted_True.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        block = pickle.load(fp)

    if second_iter and BATCH<0.1:
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi_2 = pickle.load(fp)

            dist = dist+dist_2
            gene = gene+gene_2
            combi = combi+combi_2

    block = block[:100]


list_rand_2 = []
list_orde_2 = []
list_dist_2 = []
list_mixe_2 = []
list_gene_2 = []
list_combi_2 = []
list_block_2 = []

total_number = np.minimum(np.minimum(len(dist), len(gene)), len(combi))

for j in range(np.minimum(np.minimum(len(dist), len(gene)), 
                            np.minimum(len(combi),len(block))
                            )):
    dist_tot.append(dist[j][0])
    gene_tot.append(gene[j][0])
    combi_tot.append(combi[j][0])
    if dist[j][0]< maxiter:
        list_dist_2.append(dist[j][0])
    if gene[j][0]< maxiter and gene[j][0] != 3000:
        list_gene_2.append(gene[j][0])
    if combi[j][0] < maxiter:
        list_combi_2.append(combi[j][0])
    if block[j][0] < maxiter:
        list_block_2.append(block[j][0])

    


print('Lengths: linear:{}, gene:{}, combi:{}, over:{}'.format(len(dist), len(gene), len(combi), len(block)))

# Final plotting function

list_gene = []
list_block = []
list_dist = []
list_combi = []


total_number = np.minimum(
                      np.minimum(
                             len(block),len(dist)
                             ),
                      np.minimum(
                             len(gene), len(combi)
                             )
                      )

for j in range(total_number):
    list_block.append(block[j][0])
    list_gene.append(gene[j][0])
    list_dist.append(dist[j][0])
    list_combi.append(combi[j][0])

# if len(block)<total_number:



def generating_cumulative_blocks(list_arrays,name_arrays, m, max_eval, refinement,BATCH,title,LEGEND):    
    n = len(list_arrays)
    print(n,m)
    r = np.array(range(refinement+1))*max_eval/refinement
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if n!=len(name_arrays):
        print('The names and list are not the same')
        return(-1)
    
    M = np.zeros((n,len(r)))
    
    for i in range(n):
        if len(list_arrays[i]) != m:
            print('Error for ', name_arrays[i])
        for j in range(len(r)):
            M[i,j] = np.sum(list_arrays[i]<r[j])/len(list_arrays[i])
            
                
    fig  = plt.figure()
    
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    
    # plt.plot(r,np.transpose(M[:3]),lw=2.5)
    for i in range(n):
        plt.plot(r,M[i,:],color=colors[i],lw=2.5)
    if LEGEND:
        plt.legend(name_arrays, fontsize=18, framealpha=0.4)
    plt.xlabel('Queries',fontsize=18)
    plt.ylabel('CDF',fontsize=18)
    plt.axis([0, max_eval, 0 ,1],fontsize=18)
    if saving:    
        fig.savefig(title+str(BATCH)+'.pdf',bbox_inches='tight')
    
    return M
if args.Data == 'cifar':
    saving_title=main_dir+'/Results/CIFAR/Plots/CDF'+title
else:
    saving_title=main_dir+'/Results/Imagenet/Plots/CDF'+title

# generating_cumulative_blocks([list_gene,list_dist,list_combi,list_block],
#                              ['gene','BOBY-lin','combi','BOBY-over'],
#                              total_number,15000,10000,L_inf_var[0],saving_title);

LEGEND = (args.Data == 'cifar') and BATCH==0.1
if not args.Data == 'cifar':
    maxiter=15000

generating_cumulative_blocks([list_gene,list_dist,list_combi,list_block],
                             ['GenAttack','BOBYQA','COMBI','SQUARE'],
                             total_number,maxiter,10000,L_inf_var,saving_title,LEGEND)
