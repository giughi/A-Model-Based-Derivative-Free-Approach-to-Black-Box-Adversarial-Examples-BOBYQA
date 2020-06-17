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

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))


def uploading_name(attack, args):
    """
    This returns the string of the directroy of where the attack results are saved
    """
    if attack=='boby':
        name = (main_dir + '/Results/'+str(args.Data)+'/boby_adversary_'+str(args.Adversary)+'_interpolation_block_eps_'
               +str(args.eps)+'_max_eval_3000_n_channels_3_over_over_max_f_1.3_rounding_True_subspace_attack_'
               +str(args.subspace_attack)+'_subspace_dimension_'+str(args.sub_dim)+'.txt')
    elif attack =='combi':
        name = (main_dir+'/Results/'+str(args.Data)+'/combi_adversary_'+str(args.Adversary)+'_eps_'+str(args.eps)+
                '_max_eval_3000_max_iters_1_block_size_128_batch_size_64_no_hier_False_subspace_attack_'
                +str(args.subspace_attack)+ '_subspace_dimension_'+str(args.sub_dim)+'.txt')
    elif attack == 'square':
        name = (main_dir+'/Results/'+str(args.Data)+'/square_adversary_'+str(args.Adversary)+'_eps_'+str(args.eps)+
                '_max_eval_3000_p_init_0.1_subspace_attack_'+str(args.subspace_attack)+'_subspace_dimension_'
                +str(args.sub_dim)+'.txt')
    elif attack == 'gene':
        name = (main_dir+'/Results/'+str(args.Data)+'/gene_adversary_' + str(args.Adversary) + '_eps_'+str(args.eps) +
                '_max_eval_3000_pop_size_6_mutation_rate_0.005_alpha_0.2' +
                '_resize_dim_96_adaptive_True' +
                '_subspace_attack_' + str(args.subspace_attack) +
                '_subspace_dimension_' + str(args.sub_dim) +
                '.txt')
    elif attack == 'FW':
        name = (main_dir+'/Results/'+str(args.Data)+'/FW_adversary_' + str(args.Adversary) +
                '_eps_'+str(args.eps) +
                '_max_eval_3000_att_iter_10000_grad_est_batch_size_' +
                '25_l_r_0.01_delta_0.01_beat1_0.99_sensing_type_gaussian'+
                '_subspace_attack_' + str(args.subspace_attack) +
                '_subspace_dimension_' + str(args.sub_dim) + 
                '.txt')
    return name


def import_results(attack, args, both=False):
    """
    This function uploads the results relative to input attack and wtih the required args
    """
    if both:
        args.Adversary=True

    # import the data according to the dataset
    name = uploading_name(attack,args)

    with open(name, "rb") as fp:
        uploaded_results = pickle.load(fp)

    result = []
    for i in range(len(uploaded_results)):
        result.append(uploaded_results[i][0])
    
    return result


def map_to_complete_names(list_available_attacks, both):
    """
    This function uploads the saving name of the actual attack on the image.
    """
    saving_attacks_names = []
    for attack in list_available_attacks:
        if attack=='boby':
            saving_attacks_names.append('BOBYQA')
        if attack=='combi':
            saving_attacks_names.append('Parsimonious')
        if attack=='square':
            saving_attacks_names.append('Square')
        if attack=='gene':
            saving_attacks_names.append('GenAttack')
        if attack=='FW':
            saving_attacks_names.append('Frank-Wolfe')
    return saving_attacks_names
            