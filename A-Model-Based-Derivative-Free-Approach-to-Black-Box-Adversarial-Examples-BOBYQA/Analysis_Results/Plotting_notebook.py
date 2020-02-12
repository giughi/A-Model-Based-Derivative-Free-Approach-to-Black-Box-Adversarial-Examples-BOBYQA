# plotting Setups/Plotting_notebook.py --eps=0.01 --Data=Imagenet --second_iter=True

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

from mpl_toolkits.axes_grid.inset_locator import inset_axes

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
parser.add_argument("--adv_inception",default=False, help="Using adversary results", type=bool)
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

median_rand = []
mean_rand   = []
median_orde = []
mean_orde   = []
median_dist = []
mean_dist   = []
median_mixe = []
mean_mixe   = []
median_gene = []
mean_gene   = []
median_combi = []
mean_combi = []

dist_tot = []
gene_tot = []
combi_tot = []

for i in range(len(L_inf_var)):
    dist_tot = []
    gene_tot = []
    combi_tot = []
    dist_tot_adv = []
    gene_tot_adv = []
    combi_tot_adv = []
    BATCH = L_inf_var[i]
    
    if dataset == 'MNIST':
        # Upload  the distilled attacks
        if BATCH == 0.2 or BATCH == 0.05:    
            with open(main_dir+'/Results/MNIST/distilled_dist_L_inf_mnist_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_adv = pickle.load(fp)
            with open(main_dir+'/Results/MNIST/distilled_gene_L_inf_mnist_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_adv = pickle.load(fp)
        else:
            with open(main_dir+'/Results/MNIST/distilled_dist_L_inf_mnist_'+str(BATCH)+'_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_adv = pickle.load(fp)
            with open(main_dir+'/Results/MNIST/distilled_gene_L_inf_mnist_'+str(BATCH)+'_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_adv = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/combi_L_inf_'+str(BATCH)+'_max_eval_3000_distilled_.txt',"rb") as fp:
            combi_adv = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/Square_summary_dist_L_inf_'+str(BATCH)+'_adv_True.txt',"rb") as fp:
            block_adv = pickle.load(fp)
             
    
        # upload normal attacks
        with open(main_dir+'/Results/MNIST/dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/combi_L_inf_'+str(BATCH)+'_max_eval_3000_normal.txt',"rb") as fp:
            combi = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/Square_summary_dist_L_inf_'+str(BATCH)+'.txt',"rb") as fp:
            block = pickle.load(fp)
             
    elif dataset == 'CIFAR':

        if BATCH == 0.1 or BATCH == 0.05:    
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_adv = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_adv = pickle.load(fp)
        elif BATCH == 0.02:
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_0.0225_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_adv = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_0.0225_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_adv = pickle.load(fp)
        else:
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_'+str(BATCH)+'_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_adv = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_'+str(BATCH)+'_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_adv = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_3000_distilled_.txt',"rb") as fp:
            combi_adv = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/Square_summary_dist_L_inf_'+str(BATCH)+'_adv_True.txt',"rb") as fp:
            block_adv = pickle.load(fp)
         
        

        with open(main_dir+'/Results/CIFAR/dist_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/gene_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/CONVALIDATIONcombi_L_inf_'+str(BATCH)+'_max_eval_3000_normal.txt',"rb") as fp:
            combi = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/Square_summary_dist_L_inf_'+str(BATCH)+'_adv_False.txt',"rb") as fp:
            block = pickle.load(fp)

    elif dataset == 'STL10':
        if BATCH in [0.03, 0.05, 0.1] :
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist1 = pickle.load(fp)
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist2 = pickle.load(fp)
            dist = dist1 + dist2
        else:
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist = pickle.load(fp)
        if BATCH in [0.03, 0.05, 0.1] :
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_GEN_'+str(BATCH)+'_max_eval_3000_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene1 = pickle.load(fp)
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_GEN_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene2 = pickle.load(fp)
            gene = gene1 + gene2
        else:
            with open(main_dir+'/Results/STL10/simple_attack_L_inf_GEN_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene = pickle.load(fp)
        if BATCH == 0.03:
            with open(main_dir+'/Results/STL10/combi_simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_second_NOV.txt', "rb") as fp:
                combi1 = pickle.load(fp)
            with open(main_dir+'/Results/STL10/combi_simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_2.txt', "rb") as fp:
                combi2 = pickle.load(fp)
            combi = combi1 + combi2
        elif BATCH == 0.02:
            with open(main_dir+'/Results/STL10/combi_simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_second_NOV.txt', "rb") as fp:
                combi = pickle.load(fp)
        else:
            with open(main_dir+'/Results/STL10/combi_simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_2.txt', "rb") as fp:
                combi = pickle.load(fp)
        if BATCH == 0.03:
            with open(main_dir+'/Results/STL10/BLOCK_attack_L_inf_'+str(BATCH)+'_max_eval_3000_one_round.txt', "rb") as fp:
                block = pickle.load(fp)
        elif BATCH == 0.02:
            with open(main_dir+'/Results/STL10/BLOCK_attack_L_inf_'+str(BATCH)+'_max_eval_3000_one_round_2.txt', "rb") as fp:
                block = pickle.load(fp)
        else:
            with open(main_dir+'/Results/STL10/BLOCK_attack_L_inf_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:
                block = pickle.load(fp)    
        # print(dist)
        dist = block    
        
        
    elif dataset == 'Imagenet':
        if not args.only_second:
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
                gene = gene[:100]
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
                combi = combi3 #combi1[:60] + combi2
                # dist = combi1[:60] + combi2

            if BATCH == 0.01:
                # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                #     dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi = pickle.load(fp)
            
            if second_iter and BATCH<0.1:
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist_2 = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    gene_2 = pickle.load(fp)
                with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi_2 = pickle.load(fp)
                print(second_iter,len(dist_2),len(gene_2), len(combi_2))
                dist = dist+dist_2
                gene = gene+gene_2
                combi = combi+combi_2
        else:
            if BATCH != 0.008:
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist_2 = pickle.load(fp)
            else:
                with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_2nd_50025.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    dist_2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_2nd_500_.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_2 = pickle.load(fp)
            if BATCH != 0.008:
                with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_2nd_500.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi_2 = pickle.load(fp)
            else:
                with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_function.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    combi_2 = pickle.load(fp)
            print(second_iter,len(dist_2),len(gene_2), len(combi_2))
            dist = dist_2
            gene = gene_2
            combi = combi_2

        with open(main_dir+'/Results/Imagenet/SQUA_'+str(BATCH)+'_targeted_True.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                block = pickle.load(fp)
        
        if args.adv_inception:
            with open('./Results/Imagenet/ADV_BOBY_'+str(BATCH)+ '_2nd_50050_5.txt',"rb") as fp:
                boby_adv = pickle.load(fp)
            with open('./Results/Imagenet/ADV_BOBY_'+str(BATCH)+ '_2nd_50050_5_jumping_50.txt',"rb") as fp:
                boby_adv_2 = pickle.load(fp)
                boby_adv =  boby_adv + boby_adv_2
            # with open('./Results/Imagenet/ADV_BOBY_'+str(BATCH)+ '_2nd_50050.txt',"rb") as fp:
            #     boby = pickle.load(fp)
            with open('./Results/Imagenet/ADV_COMBI_'+str(BATCH)+ '_function_jumping_0.txt',"rb") as fp:
                comb_adv = pickle.load(fp)  
        
        # print(dist)
    elif dataset == 'MiniImageNet':
        with open(main_dir+'/Results/MiniImageNet/BOBY_'+str(BATCH)+'_FINAL_linear.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/GENE_'+str(BATCH)+'_15k.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            block = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/COMBI_'+str(BATCH)+'.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            combi = pickle.load(fp)

    list_rand_2 = []
    list_orde_2 = []
    list_dist_2 = []
    list_mixe_2 = []
    list_gene_2 = []
    list_combi_2 = []
    list_block_2 = []

    list_dist_2_adv = []
    list_gene_2_adv = []
    list_combi_2_adv = []
    list_block_2_adv = []
    
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

    if dataset=='MNIST' or dataset=='CIFAR':
        total_number_adv = np.minimum(np.minimum(len(dist_adv), len(gene_adv)),
                                      np.minimum(len(combi_adv), len(block_adv)))

        for j in range(total_number_adv):
            dist_tot_adv.append(dist_adv[j][0])
            gene_tot_adv.append(gene_adv[j][0])
            combi_tot_adv.append(combi_adv[j][0])
            if dist_adv[j][0]< maxiter:
                list_dist_2_adv.append(dist_adv[j][0])
            if gene_adv[j][0]< maxiter and gene_adv[j][0] != 3000:
                list_gene_2_adv.append(gene_adv[j][0])
            if combi_adv[j][0] < maxiter:
                list_combi_2_adv.append(combi_adv[j][0])
            if block_adv[j][0] < maxiter:
                list_block_2_adv.append(block_adv[j][0])
    # median_dist.append(np.quantile(dist_tot,quant))
    # mean_dist.append(np.mean(dist))
    # median_gene.append(np.quantile(gene_tot,quant))
    # mean_gene.append(np.mean(gene))
    # median_combi.append(np.quantile(combi_tot,quant))
    # mean_combi.append(np.mean(combi))
    


print('Lengths: BOBYQA:{}, GenAttack:{}, COMBI:{}, SQUARE:{}'.format(len(dist), len(gene), len(combi), len(block)))

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



def generating_cumulative_blocks(list_arrays,name_arrays, m, max_eval, refinement,BATCH,title,legend,
                                zoom):    
    n = len(list_arrays)
    print(n,m)
    r = np.array(range(refinement+1))*max_eval/refinement
    
    fontSize = 18

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
    if legend:
        plt.legend(name_arrays,loc=1, fontsize=fontSize, framealpha=0.4)
    plt.xlabel('Queries',fontsize=fontSize)
    plt.ylabel('CDF',fontsize=fontSize)
    plt.axis([0, max_eval, 0 ,1],fontsize=fontSize)
    
    if zoom:
        # ax = fig.add_subplot(111)
        # inse_axes = inset_axes(ax, 
        #             width="50%", # width = 30% of parent_bbox
        #             height="40%",#1.0, # height : 1 inch
        #             loc=1)
        a = plt.axes([.45, .45, .4, .4],)
        dimni = 5
        for i in range(n):
            plt.plot(r,M[i,:],color=colors[i],lw=1.5)
        plt.xlabel('Queries',fontsize=fontSize-dimni)
        plt.ylabel('CDF',fontsize=fontSize-dimni)
        plt.axis([5000, max_eval, 0 ,0.19],fontsize=fontSize-dimni)
        # plt.rc('xtick',fontsize=8)
        # plt.rc('ytick',fontsize=8)
        plt.xticks(fontsize=fontSize-dimni)
        plt.yticks(fontsize=fontSize-dimni)

    if saving:    
        fig.savefig(title+str(BATCH)+'.pdf',bbox_inches='tight')
        

    return M

saving_title=main_dir+'/Results/'+dataset+'/Plots/CDF'+title

# generating_cumulative_blocks([list_gene,list_dist,list_combi,list_block],
#                              ['gene','BOBY-lin','combi','BOBY-over'],
#                              total_number,15000,10000,L_inf_var[0],saving_title);


LEGEND = (BATCH in [0.4,0.1]) or (BATCH == 0.05 and dataset == 'Imagenet') or (BATCH == 0.03 and dataset == 'MiniImageNet')

adv_inception = args.adv_inception
print('USING ADV INCEPTION', adv_inception)


if (dataset == 'Imagenet' or dataset == 'STL10' or dataset == 'MiniImageNet') and not adv_inception:
    if dataset=='STL10':
        maxiter=3000
    generating_cumulative_blocks([list_gene,list_dist,list_combi, list_block],
                                ['GenAttack','BOBYQA','COMBI','SQUARE'],
                                total_number,maxiter,10000,L_inf_var[0],saving_title,LEGEND,BATCH==0.01)
elif adv_inception:
    def check_double(l_a,line):
        found = False
        for j in range(line):
            if l_a[line][1]==l_a[j][1]:
                found = True
                if l_a[line][2]!=l_a[j][2]:
                    print('Same Initial different Target within method')
                break
        return found

    def find_row(l_ael,l_b):
        found = False
        line = None
        for j in range(len(l_b)):
            if l_ael[1]==l_b[j][1]:
                found = True
                line = j
                if l_ael[2]!=l_b[j][2]:
                    print('Same Initial different Target between methods')
                    found = False
                break
        return found, line

    def join_vectors(l_a,l_b):
        L_a = []
        L_b = []
        for line in range(len(l_a)):
            double = check_double(l_a,line)
            if not double:
                found, line_b = find_row(l_a[line],l_b)
                if found:
                    L_a.append(l_a[line])
                    L_b.append(l_b[line_b])
        return L_a, L_b

    BOBY, COMBI = join_vectors(boby_adv, comb_adv)

    def generating_cumulative_together_unbalances(list_arrays,name_arrays, m1, m2, max_eval, refinement,BATCH,
                                        title,LEGEND,zoom):    
        n = len(list_arrays)
        # m = len(list_arrays[0])
        print(m1,m2)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        r = np.array(range(int(refinement)+1))*max_eval/float(refinement)

        M = np.zeros((n,len(r)))

        for i in range(n):
            for j in range(len(r)):
                if i<4:
                    M[i,j] = np.sum(list_arrays[i][:m1]<r[j])/m1
                else:
                    M[i,j] = np.sum(list_arrays[i][:m2]<r[j])/m2
    #     M = M/m
        fig  = plt.figure()

        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)

        pl7, = plt.plot(r,M[1,:],'--',color='white',lw=2)
        pl5, = plt.plot(r,M[1,:],'--',color='white',lw=2)
        pl0, = plt.plot(r,M[0,:],lw=2)
        pl1, = plt.plot(r,M[1,:],lw=2)
        pl2, = plt.plot(r,M[2,:],lw=2)
        pl3, = plt.plot(r,M[5,:],'--',color=colors[1],lw=2)
        pl4, = plt.plot(r,M[4,:],'--',color=colors[2],lw=2)
        
        pl6, = plt.plot(r,M[3,:],color=colors[3],lw=2)
        


        if LEGEND:
    #     plt.plot(r,np.transpose(M),lw=2)
            l1 = plt.legend([pl0,pl1,pl2,pl6], ['','','',''],loc=1, fontsize=16, framealpha=0.,ncol=1,
                        bbox_to_anchor=(0.2,1))
            l2 = plt.legend([pl5,pl3,pl4,pl7], name_arrays,loc=1, fontsize=16, framealpha=0.,ncol=1, bbox_to_anchor=(0.55,1))

            plt.gca().add_artist(l1)
        plt.xlabel('Queries',fontsize=18)
        plt.ylabel('CDF',fontsize=18)
        plt.axis([0, max_eval, 0 ,1],fontsize=18)

        if LEGEND:
            plt.text(250,0.95,'Norm', fontsize=16)
            plt.text(2200,0.95,'Adv', fontsize=16)

        if zoom:
            # ax = fig.add_subplot(111)
            # inse_axes = inset_axes(ax, 
            #             width="50%", # width = 30% of parent_bbox
            #             height="40%",#1.0, # height : 1 inch
            #             loc=1)
            fontSize=18
            a = plt.axes([.45, .45, .4, .4],)
            dimni = 5
            for i in range(4):
                plt.plot(r,M[i,:],color=colors[i],lw=1.5)
            
            plt.plot(r,M[4,:],'--',color=colors[2],lw=1.5)
            plt.plot(r,M[5,:],'--',color=colors[1],lw=1.5)
            
            plt.xlabel('Queries',fontsize=fontSize-dimni)
            plt.ylabel('CDF',fontsize=fontSize-dimni)
            plt.axis([5000, max_eval, 0 ,0.19],fontsize=fontSize-dimni)
            # plt.rc('xtick',fontsize=8)
            # plt.rc('ytick',fontsize=8)
            plt.xticks(fontsize=fontSize-dimni)
            plt.yticks(fontsize=fontSize-dimni)

        fig.savefig(title+str(BATCH)+'_with_adversary.pdf',bbox_inches='tight')

        return M
    print('Doing all together')
    M = generating_cumulative_together_unbalances(list_arrays=[list_gene,list_dist,list_combi, list_block, np.array(COMBI)[:,0], np.array(BOBY)[:,0]], 
                                              name_arrays=['GenAttack','BOBYQA','COMBI','SQUARE'], 
                                              m1=total_number,m2=len(COMBI),max_eval=15000,BATCH=BATCH,title=saving_title,LEGEND=LEGEND,zoom=BATCH==0.01, refinement=1000)    
    print('Gene:', M[0,-1])
    print('BOBY:', M[1,-1])
    print('COMB:', M[2,-1])
    print('SQUA:', M[3,-1])
    print('BOBY_adv:', M[4,-1])
    print('COMB_adv:', M[5,-1])

else:




    def generating_cumulative_together(list_arrays,name_arrays, m1, m2, max_eval, refinement,BATCH,
                                       title,LEGEND):    
        n = len(list_arrays)
        # m = len(list_arrays[0])
        print(m1,m2)
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        r = np.array(range(refinement+1))*max_eval/refinement
        
        M = np.zeros((n,len(r)))
        
        for i in range(n):
            for j in range(len(r)):
                if i<3:
                    M[i,j] = np.sum(list_arrays[i][:m1]<r[j])/m1
                else:
                    M[i,j] = np.sum(list_arrays[i][:m2]<r[j])/m2
    #     M = M/m
        fig  = plt.figure()
        
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        
        pl0, = plt.plot(r,M[0,:],lw=2)
        pl1, = plt.plot(r,M[1,:],lw=2)
        pl2, = plt.plot(r,M[2,:],lw=2)
        pl3, = plt.plot(r,M[3,:],'--',color=colors[0],lw=2)
        pl4, = plt.plot(r,M[4,:],'--',color=colors[1],lw=2)
        pl5, = plt.plot(r,M[5,:],'--',color=colors[2],lw=2)
        pl6, = plt.plot(r,M[6,:],color=colors[3],lw=2)
        pl7, = plt.plot(r,M[7,:],'--',color=colors[3],lw=2)
        

        if LEGEND:
    #     plt.plot(r,np.transpose(M),lw=2)
            l1 = plt.legend([pl0,pl1,pl2,pl6], ['','','',''],loc=1, fontsize=16, framealpha=0.,ncol=1,
                           bbox_to_anchor=(0.65,0.43))
            l2 = plt.legend([pl3,pl4,pl5,pl7], name_arrays,loc=0, fontsize=16, framealpha=0.,ncol=1)
            
            plt.gca().add_artist(l1)
        plt.xlabel('Queries',fontsize=18)
        plt.ylabel('CDF',fontsize=18)
        plt.axis([0, max_eval, 0 ,1],fontsize=18)
        
        if LEGEND:
            plt.text(1380,0.4,'Norm', fontsize=16)
            plt.text(1820,0.4,'Adv', fontsize=16)

        fig.savefig(title+str(BATCH)+'_together.pdf',bbox_inches='tight')
        
        return M

    m1 = np.minimum(len(dist), np.minimum(len(gene), len(combi)))

    m2 = np.minimum(len(dist_adv),np.minimum(len(gene_adv), len(combi_adv)))

    generating_cumulative_together([list_gene_2,list_dist_2,list_combi_2,
                                    list_gene_2_adv,list_dist_2_adv,list_combi_2_adv,
                                    list_block_2,list_block_2_adv],
                                   ['GenAttack','BOBYQA','COMBI','SQUARE'],
                            m1,m2,3000,10000,L_inf_var[0],saving_title, LEGEND)
