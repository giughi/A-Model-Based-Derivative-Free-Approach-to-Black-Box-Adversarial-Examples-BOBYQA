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
parser.add_argument("--Data", default='Imagenet', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR','STL10','Imagenet','MiniImageNet'")
parser.add_argument("--title", default='', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument("--Adversary", default=False, help="Boolean for plotting adversarial attacks too", type=bool)
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=10000, help="Maximum iterations allowed", type=int)
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

dataset = args.Data 
adversary = args.Adversary
saving = args.save
plot_type = args.plot_type
quant = args.quantile
maxiter = args.max_iter
title = args.title


median_dist = []
median_gene = []
median_combi = []
median_dist_d = []
median_gene_d = []
median_combi_d = []

dist_tot = []
gene_tot = []
combi_tot = []
dist_tot_d = []
gene_tot_d = []
combi_tot_d = []

if dataset == 'MNIST':
    L_inf_var = [0.4,0.3,0.2]
    maxiter = 3000
elif dataset == 'CIFAR':
    L_inf_var = [0.1,0.05,0.02]
    maxiter = 3000
elif dataset == 'STL10':
    L_inf_var = [0.4,0.3,0.2,0.1]
elif dataset == 'Imagenet':
    L_inf_var = [0.1,0.05,0.03,0.02]
elif dataset == 'MiniImageNet':
    L_inf_var = [0.05,0.02,0.01,0.05]


for i in range(len(L_inf_var)):
    dist_tot = []
    gene_tot = []
    combi_tot = []
    BATCH = L_inf_var[i]
    
    if dataset == 'MNIST':
        print('uploading MNIST')
        # Upload  the distilled attacks
        if BATCH == 0.2 or BATCH == 0.05:    
            with open(main_dir+'/Results/MNIST/distilled_dist_L_inf_mnist_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_d = pickle.load(fp)
            with open(main_dir+'/Results/MNIST/distilled_gene_L_inf_mnist_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_d = pickle.load(fp)
        else:
            with open(main_dir+'/Results/MNIST/distilled_dist_L_inf_mnist_'+str(BATCH)+'_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_d = pickle.load(fp)
            with open(main_dir+'/Results/MNIST/distilled_gene_L_inf_mnist_'+str(BATCH)+'_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_d = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/combi_L_inf_'+str(BATCH)+'_max_eval_3000_distilled_.txt',"rb") as fp:
            combi_d = pickle.load(fp)
        
        with open(main_dir+'/Results/MNIST/dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/MNIST/combi_L_inf_'+str(BATCH)+'_max_eval_3000_normal.txt',"rb") as fp:
            combi = pickle.load(fp)

    elif dataset == 'CIFAR':
        if BATCH == 0.1 or BATCH == 0.05:    
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_d = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_'+str(BATCH)+'_2 (1).txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_d = pickle.load(fp)
        elif BATCH == 0.02:
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_0.0225_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_d = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_0.0225_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_d = pickle.load(fp)
        else:
            with open(main_dir+'/Results/CIFAR/distilled_dist_L_inf_cifar10_'+str(BATCH)+'_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist_d = pickle.load(fp)
            with open(main_dir+'/Results/CIFAR/distilled_gene_L_inf_cifar10_'+str(BATCH)+'_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene_d = pickle.load(fp)
        
        
        with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_3000_distilled_.txt',"rb") as fp:
            combi_d = pickle.load(fp)
        
        with open(main_dir+'/Results/CIFAR/dist_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/gene_L_inf_cifar10_'+str(BATCH)+'_refined.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/CIFAR/combi_L_inf_'+str(BATCH)+'_max_eval_3000_normal.txt',"rb") as fp:
            combi = pickle.load(fp)
                
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
        else:
            with open(main_dir+'/Results/STL10/combi_simple_attack_L_inf_'+str(BATCH)+'_max_eval_3000_2.txt', "rb") as fp:
                combi = pickle.load(fp)
        if BATCH == 0.03:
            with open(main_dir+'/Results/STL10/BLOCK_attack_L_inf_'+str(BATCH)+'_max_eval_3000_one_round.txt', "rb") as fp:
                block = pickle.load(fp)
        else:
            with open(main_dir+'/Results/STL10/BLOCK_attack_L_inf_'+str(BATCH)+'_max_eval_3000.txt', "rb") as fp:
                block = pickle.load(fp)        
        
    elif dataset == 'Imagenet':
        if  BATCH == 0.02:
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over_3.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                block = pickle.load(fp)
        elif BATCH == 0.1:
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear_2.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                block = pickle.load(fp)
        else:
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_linear.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                dist = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                block = pickle.load(fp)
        if BATCH == 0.1:
            with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k_2.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                gene = pickle.load(fp)
        else:
            with open(main_dir+'/Results/Imagenet/GENE_'+str(BATCH)+'_15k.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                    gene = pickle.load(fp)
        # with open(main_dir+'/Results/Imagenet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
        #     block = pickle.load(fp)
        with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi = pickle.load(fp)
        if BATCH == 0.02:
            with open(main_dir+'/Results/Imagenet/COMBI_'+str(BATCH)+'_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi1 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/COMBI_0.02starting_60_FINAL.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi2 = pickle.load(fp)
            with open(main_dir+'/Results/Imagenet/COMBI_0.02_1_iter.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
                combi3 = pickle.load(fp)
            combi = combi1[:60] + combi2

            
    elif dataset == 'MiniImageNet':
        with open(main_dir+'/Results/MiniImageNet/BOBY_'+str(BATCH)+'_FINAL_linear.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/GENE_'+str(BATCH)+'_15k.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            gene = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/BOBY_'+str(BATCH)+'_FINAL_over.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            block = pickle.load(fp)
        with open(main_dir+'/Results/MiniImageNet/COMBI_'+str(BATCH)+'.txt', "rb") as fp:#('gene_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            combi = pickle.load(fp)

    
    total_number = np.minimum(np.minimum(len(dist), len(gene)), len(combi))
    if dataset == 'MNIST' or dataset == 'CIFAR':
        total_number = np.minimum(np.minimum(np.minimum(len(dist), len(gene)), 
                                  np.minimum(len(combi), len(combi_d))),
                                  np.minimum(len(dist_d), len(gene_d)))
    
    for j in range(total_number):
        if dist[j][0]>maxiter:
            dist_tot.append(maxiter)
        else:
            dist_tot.append(dist[j][0])
        if gene[j][0]>maxiter:
            gene_tot.append(maxiter)
        else:
            gene_tot.append(gene[j][0])
        if combi[j][0]>maxiter:
            combi_tot.append(maxiter)
        else:
            combi_tot.append(combi[j][0])


        if dataset == 'MNIST' or dataset == 'CIFAR':
            if dist_d[j][0]>maxiter:
                dist_tot_d.append(maxiter)
            else:
                dist_tot_d.append(dist_d[j][0])
            if gene_d[j][0]>maxiter:
                gene_tot_d.append(maxiter)
            else:
                gene_tot_d.append(gene_d[j][0])
            if combi_d[j][0]>maxiter:
                combi_tot_d.append(maxiter)
            else:
                combi_tot_d.append(combi_d[j][0])


    median_dist.append(np.quantile(dist_tot,quant))
    median_gene.append(np.quantile(gene_tot,quant))
    median_combi.append(np.quantile(combi_tot,quant))
    if dataset == 'MNIST' or dataset == 'CIFAR':
        median_dist_d.append(np.quantile(dist_tot_d,quant))
        median_gene_d.append(np.quantile(gene_tot_d,quant))
        median_combi_d.append(np.quantile(combi_tot_d,quant))


print('Lengths: linear:{}, gene:{}, combi:{}'.format(len(dist), len(gene), len(combi)))


def generating_quantiles(list_arrays,name_arrays, m, BATCH,title):    
    n = len(list_arrays)
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # if n!=len(name_arrays):
    #     print('The names and list are not the same')
    #     return(-1)
    
    fig  = plt.figure()

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    
    pl0, = plt.plot(BATCH,list_arrays[0],lw=2)
    pl1, = plt.plot(BATCH,list_arrays[1],lw=2)
    pl2, = plt.plot(BATCH,list_arrays[2],lw=2)
    if dataset == 'MNIST' or dataset == 'CIFAR':
        pl3, = plt.plot(BATCH,list_arrays[3],'--',color=colors[0],lw=2)
        pl4, = plt.plot(BATCH,list_arrays[4],'--',color=colors[1],lw=2)
        pl5, = plt.plot(BATCH,list_arrays[5],'--',color=colors[2],lw=2)
    
    plt.xlabel('$\epsilon_{\infty}$',fontsize=16)
    plt.ylabel('# queries',fontsize=16)
    # plt.axis([0, max_eval, 0 ,1],fontsize=16)
    # plt.legend(name_arrays,loc=4, fontsize=16, framealpha=0.4)

    l1 = plt.legend([pl0,pl1,pl2], ['','',''],loc=0, fontsize=16, framealpha=0.,ncol=1,
                   bbox_to_anchor=(0.53,0.66))
    l2 = plt.legend([pl3,pl4,pl5], name_arrays,loc=0, fontsize=16, framealpha=0.,ncol=1)
    
    plt.gca().add_artist(l1)
    
    plt.text(0.31,1650,'Norm', fontsize=16)
    plt.text(0.34,1650,'Adv', fontsize=16)

    fig.savefig(title+str(BATCH)+'_together.pdf',bbox_inches='tight')
    
    
saving_title=main_dir+'/Results/'+dataset+'/Plots/Quantiles'+str(quant)+title

data = [median_dist, median_gene, median_combi]
datasets = ['GenA', 'BOBY', 'COMBI']
if dataset == 'MNIST' or dataset == 'CIFAR':
    data = [median_dist, median_gene, median_combi,median_dist_d, median_gene_d, median_combi_d]
    datasets = ['GenA', 'BOBY', 'COMBI']
    
generating_quantiles(data,
                     datasets,
                     maxiter,
                     L_inf_var,
                     saving_title);