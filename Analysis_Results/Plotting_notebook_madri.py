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

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')



parser = argparse.ArgumentParser()
# parser.add_argument("eps", default=0.1, help="Energy of the pertubation", type=float)
parser.add_argument("--eps", default=[0.1], nargs='+', type=float, help="Energy of the pertubation")
parser.add_argument("--Data", default='Imagenet', help="Dataset that we want to attack. At the moment we have:'MNIST','CIFAR','STL10','Imagenet','MiniImageNet'")
parser.add_argument("--title", default='', help="This will be associated to the image title")
parser.add_argument("--plot_type", default='CDF', help="What graph we generate; `CDF` or `quantile`")
parser.add_argument("--save",default=True, help="Boolean to save the result", type=bool)
parser.add_argument('--Adversary', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--subspace_attack', type=str_to_bool, nargs='?', const=True, default=False)

# parser.add_argument("--Adversary", default=False, action='store_true', help="Boolean for plotting adversarial attacks too")
parser.add_argument("--quantile", default=0.5, help="If in `quantile` option, it says which quantile to plot", type=float)
parser.add_argument("--max_iter", default=15000, help="Maximum iterations allowed", type=int)
parser.add_argument("--sub_dim", default=1000, help="Subdomain Dimension", type=int)
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

if not args.subspace_attack:
    args.sub_dim=None
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
    block_tot = []
    FW_tot = []
    dist_tot_adv = []
    gene_tot_adv = []
    combi_tot_adv = []
    BATCH = L_inf_var[i]
    print('ADversary==',args.Adversary)
    if args.Data=='CIFAR':
        # finding intervals of the data
        if not args.Adversary:
            if BATCH==0.02:
                interval = [int(i) for i in range(1146)]
            elif BATCH==0.05:
                interval = np.arange(1146,2292)
            elif BATCH==0.1:
                interval = np.arange(2292, 3438)
            elif BATCH==0.15:
                interval = np.arange(3438,4580)
        else:
            if BATCH==0.02:
                interval = [int(i) for i in range(1092)]
            elif BATCH==0.05:
                interval = np.arange(1092,2101)
            elif BATCH==0.1:
                interval = np.arange(2184, 3276)
            elif BATCH==0.15:
                interval = np.arange(3276, 4368)

        with open(main_dir+'/Results/CIFAR/boby_adversary_'+str(args.Adversary)+'_interpolation_block_eps_'
                 +str(BATCH)+'_max_eval_3000_n_channels_3_over_over_max_f_1.3_rounding_True_subspace_attack_'
                 +str(args.subspace_attack)+'_subspace_dimension_'+str(args.sub_dim)+'.txt', "rb") as fp:
            dist_copy = pickle.load(fp)
            dist = []
            if not args.Adversary and BATCH >= 0.02:
                for i in interval:
                    dist.append(dist_copy[i])
            elif args.Adversary and BATCH >= 0.02:
                for i in interval:
                    dist.append(dist_copy[i])
            else:
                dist=dist_copy
        
        with open(main_dir+'/Results/CIFAR/combi_adversary_'+str(args.Adversary)+'_eps_'+str(BATCH)+
            '_max_eval_3000_max_iters_1_block_size_128_batch_size_64_no_hier_False_subspace_attack_'
            +str(args.subspace_attack)+ '_subspace_dimension_'+str(args.sub_dim)+'.txt',"rb") as fp:
            combi_copy = pickle.load(fp)
            combi = []
            if not args.Adversary and BATCH >= 0.02:
                for i in interval:
                    combi.append(combi_copy[i])
            elif args.Adversary and BATCH >= 0.02 and BATCH <=0.05:
                for i in interval:
                    combi.append(combi_copy[i])
            else:
                combi = combi_copy
            
        with open(main_dir+'/Results/CIFAR/square_adversary_'+str(args.Adversary)+'_eps_'+str(BATCH)+
            '_max_eval_3000_p_init_0.1_subspace_attack_'+str(args.subspace_attack)+'_subspace_dimension_'
            +str(args.sub_dim)+'.txt',"rb") as fp:
            block_copy = pickle.load(fp)
            block = []
            if not args.Adversary and BATCH >= 0.02:
                for i in interval:
                    block.append(block_copy[i])
            elif args.Adversary and BATCH >= 0.02:
                for i in interval:
                    block.append(block_copy[i])
            else:
                block = block_copy

        with open(main_dir+'/Results/CIFAR/gene_adversary_' + str(args.Adversary) + '_eps_'+str(BATCH) +
                                '_max_eval_3000_pop_size_6_mutation_rate_0.005_alpha_0.2' +
                                '_resize_dim_96_adaptive_True' +
                                '_subspace_attack_' + str(args.subspace_attack) +
                                '_subspace_dimension_' + str(args.sub_dim) +
                                '.txt',"rb") as fp:
            gene_copy = pickle.load(fp)
            gene = []
            # if not args.Adversary:
            #     for i in interval:
            #         gene.append(block_copy[i])
            # else:
            gene = gene_copy

        with open(main_dir+'/Results/CIFAR/FW_adversary_' + str(args.Adversary) +
                                '_eps_'+str(BATCH) +
                                '_max_eval_3000_att_iter_10000_grad_est_batch_size_' +
                                '25_l_r_0.01_delta_0.01_beat1_0.99_sensing_type_gaussian'+
                                '_subspace_attack_' + str(args.subspace_attack) +
                                '_subspace_dimension_' + str(args.sub_dim) + 
                                '.txt',"rb") as fp:
            FW_copy = pickle.load(fp)
            FW = []
            if not args.Adversary and BATCH >= 0.02:
                for i in interval:
                    FW.append(FW_copy[i])
            elif args.Adversary and BATCH >= 0.02:
                for i in interval:
                    FW.append(FW_copy[i])
            else:
                FW = FW_copy

    elif args.Data=='Imagenet':
        with open(main_dir+'/Results/Imagenet/boby_adversary_'+str(args.Adversary)+'_interpolation_block_eps_'
                  +str(BATCH)+'_max_eval_15000_n_channels_3_over_over_max_f_1.3_rounding_True_subspace_attack_'
                  +str(args.subspace_attack)+'_subspace_dimension_'+str(args.sub_dim)+'.txt', "rb") as fp:#('dist_L_inf_'+str(BATCH)+'.txt', "rb") as fp:   # Unpickling
            dist = pickle.load(fp)
        with open(main_dir+'/Results/Imagenet/combi_adversary_'+str(args.Adversary)+'_eps_'+str(BATCH)+
                 '_max_eval_15000_max_iters_1_block_size_128_batch_size_64_no_hier_False_subspace_attack_'
                 +str(args.subspace_attack)+'_subspace_dimension_'+str(args.sub_dim)+'.txt',"rb") as fp:
            combi = pickle.load(fp)
        # args.sub_dim=1000
        with open(main_dir+'/Results/Imagenet/square_adversary_'+str(args.Adversary)+'_eps_'+str(BATCH)+
                  '_max_eval_15000_p_init_0.1_subspace_attack_'
                 +str(args.subspace_attack)+'_subspace_dimension_'+str(args.sub_dim)+'.txt',"rb") as fp:
            block = pickle.load(fp)
        FW = block
        gene = block

    list_dist_2 = []
    list_combi_2 = []
    list_block_2 = []
    list_gene_2 = []
    list_FW_2 = []

    total_number = np.minimum(np.minimum(len(dist), len(block)), 
                              np.minimum(np.minimum(len(gene), len(FW)), 
                                         len(combi)))

    print('TOTAL NUMBER', total_number)
    for j in range(total_number):
        dist_tot.append(dist[j][0])
        block_tot.append(block[j][0])
        combi_tot.append(combi[j][0])
        gene_tot.append(gene[j][0])
        FW_tot.append(FW[j][0])
        if dist[j][0]< maxiter:
            list_dist_2.append(dist[j][0])
        if gene[j][0]< maxiter and gene[j][0] != 3000:
            list_gene_2.append(gene[j][0])
        if combi[j][0] < maxiter:
            list_combi_2.append(combi[j][0])
        if block[j][0] < maxiter:
            list_block_2.append(block[j][0])
        if FW[j][0] < maxiter:
            list_FW_2.append(FW[j][0])



print('Lengths: BOBYQA:{}, COMBI:{}, SQUARE:{}, GENE:{}, FW:{}.'
        .format(len(dist), len(combi), len(block), len(gene), len(FW)))

# Final plotting function

list_gene = []
list_block = []
list_dist = []
list_combi = []
list_FW = []

# total_number = np.minimum(
#                       np.minimum(
#                              len(block),len(dist)
#                              ),
#                       np.minimum(
#                              len(combi), len(combi)
#                              )
#                       )

for j in range(total_number):
    list_block.append(block[j][0])
    list_gene.append(gene[j][0])
    list_dist.append(dist[j][0])
    list_combi.append(combi[j][0])
    list_FW.append(FW[j][0])

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

saving_title=main_dir+'/Results/'+str(args.Data)+'/Plots/CDF_adversary'+str(args.Adversary) + title + '_' + str(args.sub_dim)

if args.Data == 'CIFAR':
    generating_cumulative_blocks([list_dist,list_combi,list_block, list_gene, list_FW],
                             ['BOBY','combi','Square','Gene', 'FW'],
                             total_number,3000,1000,L_inf_var[0],saving_title, True, False);
else:
    generating_cumulative_blocks([list_dist,list_combi,list_block],
                             ['BOBY','combi','Square'],
                             total_number,15000,1000,L_inf_var[0],saving_title, True, False);


# LEGEND = (BATCH in [0.4,0.1]) or (BATCH == 0.05 and dataset == 'Imagenet') or (BATCH == 0.03 and dataset == 'MiniImageNet')

# adv_inception = args.adv_inception
# print('USING ADV INCEPTION', adv_inception)


# # if (dataset == 'Imagenet') and not adv_inception:
    
#     generating_cumulative_blocks([list_gene,list_dist,list_combi, list_block],
#                                 ['GenAttack','BOBYQA','COMBI','SQUARE'],
#                                 total_number,maxiter,10000,L_inf_var[0],saving_title,LEGEND,BATCH==0.01)
# elif adv_inception:
#     def check_double(l_a,line):
#         found = False
#         for j in range(line):
#             if l_a[line][1]==l_a[j][1]:
#                 found = True
#                 if l_a[line][2]!=l_a[j][2]:
#                     print('Same Initial different Target within method')
#                 break
#         return found

#     def find_row(l_ael,l_b):
#         found = False
#         line = None
#         for j in range(len(l_b)):
#             if l_ael[1]==l_b[j][1]:
#                 found = True
#                 line = j
#                 if l_ael[2]!=l_b[j][2]:
#                     print('Same Initial different Target between methods')
#                     found = False
#                 break
#         return found, line

#     def join_vectors(l_a,l_b):
#         L_a = []
#         L_b = []
#         for line in range(len(l_a)):
#             double = check_double(l_a,line)
#             if not double:
#                 found, line_b = find_row(l_a[line],l_b)
#                 if found:
#                     L_a.append(l_a[line])
#                     L_b.append(l_b[line_b])
#         return L_a, L_b

#     BOBY, COMBI = join_vectors(boby_adv, comb_adv)

#     def generating_cumulative_together_unbalances(list_arrays,name_arrays, m1, m2, max_eval, refinement,BATCH,
#                                         title,LEGEND,zoom):    
#         n = len(list_arrays)
#         # m = len(list_arrays[0])
#         print(m1,m2)

#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']

#         r = np.array(range(int(refinement)+1))*max_eval/float(refinement)

#         M = np.zeros((n,len(r)))

#         for i in range(n):
#             for j in range(len(r)):
#                 if i<4:
#                     M[i,j] = np.sum(list_arrays[i][:m1]<r[j])/m1
#                 else:
#                     M[i,j] = np.sum(list_arrays[i][:m2]<r[j])/m2
#     #     M = M/m
#         fig  = plt.figure()

#         plt.rc('xtick',labelsize=16)
#         plt.rc('ytick',labelsize=16)

#         pl7, = plt.plot(r,M[1,:],'--',color='white',lw=2)
#         pl5, = plt.plot(r,M[1,:],'--',color='white',lw=2)
#         pl0, = plt.plot(r,M[0,:],lw=2)
#         pl1, = plt.plot(r,M[1,:],lw=2)
#         pl2, = plt.plot(r,M[2,:],lw=2)
#         pl3, = plt.plot(r,M[5,:],'--',color=colors[1],lw=2)
#         pl4, = plt.plot(r,M[4,:],'--',color=colors[2],lw=2)
        
#         pl6, = plt.plot(r,M[3,:],color=colors[3],lw=2)
        


#         if LEGEND:
#     #     plt.plot(r,np.transpose(M),lw=2)
#             l1 = plt.legend([pl0,pl1,pl2,pl6], ['','','',''],loc=1, fontsize=16, framealpha=0.,ncol=1,
#                         bbox_to_anchor=(0.2,1))
#             l2 = plt.legend([pl5,pl3,pl4,pl7], name_arrays,loc=1, fontsize=16, framealpha=0.,ncol=1, bbox_to_anchor=(0.55,1))

#             plt.gca().add_artist(l1)
#         plt.xlabel('Queries',fontsize=18)
#         plt.ylabel('CDF',fontsize=18)
#         plt.axis([0, max_eval, 0 ,1],fontsize=18)

#         if LEGEND:
#             plt.text(250,0.95,'Norm', fontsize=16)
#             plt.text(2200,0.95,'Adv', fontsize=16)

#         if zoom:
#             # ax = fig.add_subplot(111)
#             # inse_axes = inset_axes(ax, 
#             #             width="50%", # width = 30% of parent_bbox
#             #             height="40%",#1.0, # height : 1 inch
#             #             loc=1)
#             fontSize=18
#             a = plt.axes([.45, .45, .4, .4],)
#             dimni = 5
#             for i in range(4):
#                 plt.plot(r,M[i,:],color=colors[i],lw=1.5)
            
#             plt.plot(r,M[4,:],'--',color=colors[2],lw=1.5)
#             plt.plot(r,M[5,:],'--',color=colors[1],lw=1.5)
            
#             plt.xlabel('Queries',fontsize=fontSize-dimni)
#             plt.ylabel('CDF',fontsize=fontSize-dimni)
#             plt.axis([5000, max_eval, 0 ,0.19],fontsize=fontSize-dimni)
#             # plt.rc('xtick',fontsize=8)
#             # plt.rc('ytick',fontsize=8)
#             plt.xticks(fontsize=fontSize-dimni)
#             plt.yticks(fontsize=fontSize-dimni)

#         fig.savefig(title+str(BATCH)+'_with_adversary.pdf',bbox_inches='tight')

#         return M
#     print('Doing all together')
#     M = generating_cumulative_together_unbalances(list_arrays=[list_gene,list_dist,list_combi, list_block, np.array(COMBI)[:,0], np.array(BOBY)[:,0]], 
#                                               name_arrays=['GenAttack','BOBYQA','COMBI','SQUARE'], 
#                                               m1=total_number,m2=len(COMBI),max_eval=15000,BATCH=BATCH,title=saving_title,LEGEND=LEGEND,zoom=BATCH==0.01, refinement=1000)    
#     print('Gene:', M[0,-1])
#     print('BOBY:', M[1,-1])
#     print('COMB:', M[2,-1])
#     print('SQUA:', M[3,-1])
#     print('BOBY_adv:', M[4,-1])
#     print('COMB_adv:', M[5,-1])

# else:




#     def generating_cumulative_together(list_arrays,name_arrays, m1, m2, max_eval, refinement,BATCH,
#                                        title,LEGEND):    
#         n = len(list_arrays)
#         # m = len(list_arrays[0])
#         print(m1,m2)
        
#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']
        
#         r = np.array(range(refinement+1))*max_eval/refinement
        
#         M = np.zeros((n,len(r)))
        
#         for i in range(n):
#             for j in range(len(r)):
#                 if i<3:
#                     M[i,j] = np.sum(list_arrays[i][:m1]<r[j])/m1
#                 else:
#                     M[i,j] = np.sum(list_arrays[i][:m2]<r[j])/m2
#     #     M = M/m
#         fig  = plt.figure()
        
#         plt.rc('xtick',labelsize=16)
#         plt.rc('ytick',labelsize=16)
        
#         pl0, = plt.plot(r,M[0,:],lw=2)
#         pl1, = plt.plot(r,M[1,:],lw=2)
#         pl2, = plt.plot(r,M[2,:],lw=2)
#         pl3, = plt.plot(r,M[3,:],'--',color=colors[0],lw=2)
#         pl4, = plt.plot(r,M[4,:],'--',color=colors[1],lw=2)
#         pl5, = plt.plot(r,M[5,:],'--',color=colors[2],lw=2)
#         pl6, = plt.plot(r,M[6,:],color=colors[3],lw=2)
#         pl7, = plt.plot(r,M[7,:],'--',color=colors[3],lw=2)
        

#         if LEGEND:
#     #     plt.plot(r,np.transpose(M),lw=2)
#             l1 = plt.legend([pl0,pl1,pl2,pl6], ['','','',''],loc=1, fontsize=16, framealpha=0.,ncol=1,
#                            bbox_to_anchor=(0.65,0.43))
#             l2 = plt.legend([pl3,pl4,pl5,pl7], name_arrays,loc=0, fontsize=16, framealpha=0.,ncol=1)
            
#             plt.gca().add_artist(l1)
#         plt.xlabel('Queries',fontsize=18)
#         plt.ylabel('CDF',fontsize=18)
#         plt.axis([0, max_eval, 0 ,1],fontsize=18)
        
#         if LEGEND:
#             plt.text(1380,0.4,'Norm', fontsize=16)
#             plt.text(1820,0.4,'Adv', fontsize=16)

#         fig.savefig(title+str(BATCH)+'_together.pdf',bbox_inches='tight')
        
#         return M

#     m1 = np.minimum(len(dist), np.minimum(len(gene), len(combi)))

#     m2 = np.minimum(len(dist_adv),np.minimum(len(gene_adv), len(combi_adv)))

#     generating_cumulative_together([list_gene_2,list_dist_2,list_combi_2,
#                                     list_gene_2_adv,list_dist_2_adv,list_combi_2_adv,
#                                     list_block_2,list_block_2_adv],
#                                    ['GenAttack','BOBYQA','COMBI','SQUARE'],
#                             m1,m2,3000,10000,L_inf_var[0],saving_title, LEGEND)
