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
parser.add_argument("--title", default='', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--Adversary", default=False, help="Boolean for plotting adversarial attacks too", type=bool)
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=15000, help="Maximum iterations allowed", type=int)
parser.add_argument("--second_iter",default=False, help="Loading results from second round of attacks", type=bool)
parser.add_argument("--only_second",default=False, help="Loading results from second round of attacks", type=bool)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

dataset = 'Imagenet'
L_inf_var = [0.1,0.05,0.03,0.02,0.01, 0.008]#, 0.007, 0.005]
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


dist_tot = []
gene_tot = []
combi_tot = []

for i in range(len(L_inf_var)):
    dist_tot = []
    gene_tot = []
    combi_tot = []
    BATCH = L_inf_var[i]
    print(BATCH)
    # if args.second_iter and BATCH<0.1:
    #     with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    #         dist_2 = pickle.load(fp)
    #     with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    #         gene_2 = pickle.load(fp)
    #     with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
    #         combi_2 = pickle.load(fp)

    if BATCH not in [0.008, 0.005, 0.007]:

        if (not args.only_second) or BATCH==0.1: 

            if  BATCH == 0.02:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over_3.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
            elif BATCH == 0.1:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
            elif BATCH == 0.01:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
            else:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
            if BATCH == 0.1:
                with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    gene = pickle.load(fp)
            elif BATCH == 0.02:
                    with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k_3.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                        gene = pickle.load(fp)
                    with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k_3_last30.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                        gene2 = pickle.load(fp)
                    gene = gene+gene2
                    gene = gene
            else:
                with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    gene = pickle.load(fp)
            # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            #     block = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi = pickle.load(fp)
            if BATCH == 0.02:
                # with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     combi1 = pickle.load(fp)
                # with open(main_dir+'/Results/Imagenet/COMBI_0.02starting_60_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     combi2 = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/COMBI_0.02_1_iter.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi3 = pickle.load(fp)
                combi = combi3#[:60] + combi2

            if BATCH == 0.01:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi = pickle.load(fp)

            total_number = np.minimum(np.minimum(len(dist), len(gene)), len(combi))
            dist = dist[:total_number]
            combi = combi[:total_number]
            gene = gene[:total_number]

        if args.second_iter and BATCH<0.1:
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_50025.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi_2 = pickle.load(fp)
            
            if args.only_second:
                dist = dist_2
                gene = gene_2
                combi = combi_2
            else:
                dist = dist+dist_2
                gene = gene+gene_2
                combi = combi+combi_2

    elif BATCH == 0.008:
        with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_50025.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_function.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            combi = pickle.load(fp)

    elif BATCH == 0.007:
        with open('Results/Imagenet/BOBY_'+str(eps)+'_2nd_50025.txt', "rb") as fp:
            boby = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.007_function_jumping_0.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi1 = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.007_function_jumping_20.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi2 = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.007_function_jumping_40.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi3 = pickle.load(fp)
        combi = combi1+combi2+combi3
    elif BATCH==0.005:
        with open('Results/Imagenet/BOBY_'+str(eps)+'_2nd_50025.txt', "rb") as fp:
            boby = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.005_function.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi1 = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.005_2nd_500.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi2 = pickle.load(fp)
        with open('Results/Imagenet/COMBI_0.005_function_jumping_50.txt', "rb") as fp:#open('COMBI_0.008_function.txt', "rb") as fp:
            combi3 = pickle.load(fp)
        combi = combi1+combi2+combi3

    
    block = combi
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
    
    SR_b.append(len(list_dist_2)/len(dist_tot))
    SR_g.append(len(list_gene_2)/len(gene_tot))
    SR_c.append(len(list_combi_2)/len(combi_tot))


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

saving_title=main_dir+'/Results/'+dataset+'/Plots/Energy_performance'+title

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rc('text',usetex = True)
                
fig  = plt.figure()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# plt.plot(r,np.transpose(M[:3]),lw=2.5)
plt.semilogx(L_inf_var[1:],SR_g[1:],lw=2.5,linestyle='-', marker='o')
plt.semilogx(L_inf_var[1:],SR_b[1:],lw=2.5,linestyle='-', marker='o')
plt.semilogx(L_inf_var[1:],SR_c[1:],lw=2.5,linestyle='-', marker='o')
plt.legend(['GenAttack','BOBYQA','COMBI'], fontsize=18, framealpha=0.4)
plt.xlabel(r"$\epsilon_\infty$",fontsize=18)
plt.ylabel('SR',fontsize=18)
plt.axis(fontsize=18)

fig.savefig(saving_title+str(BATCH)+'.pdf',bbox_inches='tight')
    
print('Gen', SR_g)
print('COM', SR_c)
print('BOB', SR_b)


