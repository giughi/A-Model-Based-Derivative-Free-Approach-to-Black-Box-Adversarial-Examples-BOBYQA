# coding: utf-8

# # BOBYQA Attack Procedure

# With this file we aim to generate the function that allows to attack a given image tensor

from __future__ import print_function

import sys
# import os
import tensorflow as tf
import numpy as np
# import scipy.misc
# from numba import jit
# import math
import time

#Bobyqa
import pybobyqa
import pandas as pd

import progressbar
from time import sleep
# import matplotlib.pyplot as plt

# Initialisation Coefficients

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be


class Objfun(object):
    def __init__(self, objfun):
        self._objfun = objfun
        self.nf = 0
        self.xs = []
        self.fs = []

    def __call__(self, x):
        self.nf += 1
        self.xs.append(x.copy())
        f = self._objfun(x)
        self.fs.append(f)
        return f

    def get_summary(self, with_xs=False):
        results = {}
        if with_xs:
            results['xvals'] = self.xs
        results['fvals'] = self.fs
        results['nf'] = self.nf
        results['neval'] = np.arange(1, self.nf+1)  # start from 1
        return pd.DataFrame.from_dict(results)

    def reset(self):
        self.nf = 0
        self.xs = []
        self.fs = []


def vec2modMatRand(c, indice, var, RandMatr, depend, b, a):
    temp = var.copy()
    n = len(indice)
    # print('shape of temp',temp.shape)
    # print('shape of rand',RandMatr.shape)
    # count = 0
    for i in range(n):
        indx = indice[i]
        # count+= len(RandMatr[depend == indx])
        # print('temp_size = ', len(temp[depend == indx])/temp.size)
        temp[depend == indx] += c[i]*RandMatr[depend == indx]
    # we have to clip the values to the boundaries
    # print('c',c)
    # print('total pixels', count/var.size)
    # print('nonzero in temp',np.count_nonzero(temp)/var.size)
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    temp = temp.reshape(var.shape)
    # print('nonzero in temp clipped', np.count_nonzero(temp)/var.size)
    return temp


def vec2modMatRand2(c, indice, var, RandMatr, depend, b, a):
    temp = var.copy().reshape(-1, )
    n = len(indice)
    for i in range(n):
        indices = finding_indices(depend, indice[i]).reshape(-1, )
        # print(indices)#len(RandMatr.reshape(-1, )[indices]))
        # print('len of indices is', len(indices))
        # count+= len(RandMatr[depend == indx])
        # print('temp_size = ', len(temp[depend == indx])/temp.size)
        temp[indices] += c[i]*RandMatr.reshape(-1, )[indices]
    # we have to clip the values to the boundaries
    # print('c',c)
    # print('total pixels', count/var.size)
    # print('nonzero in temp',np.count_nonzero(temp)/var.size)
    temp = np.minimum(b.reshape(-1, ), temp.reshape(-1, ))
    temp = np.maximum(a.reshape(-1, ), temp)
    temp = temp.reshape(var.shape)
    # print('nonzero in temp clipped', np.count_nonzero(temp)/var.size)
    return temp


def vec2mod(c, indice, var):
    # returns the tensor whose element in indice are var + c
    temp = var.copy()
   
    n = len(indice)
    for i in range(n):
        temp.reshape(-1)[indice[i]] += c[i]
    return temp

#########################################################
# Functions related to the optimal sampling of an image #
#########################################################


def find_neighbours(r, c, k, n, m, R):
    # This computes the neihgbours of a pixels (r,c,k) in an image R^(n,m,R)
    # Note: We never consider differnt layers of the RGB
    neighbours = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ((r+i) >= 0) and ((r+i) < n):
                if ((c+j) >= 0) and ((c+j) < m):
                    if not((i, j) == (0, 0)):
                        neighbours.append([0,r+i,c+j,k])
    return neighbours


def get_variation(img, neighbours):
    list_val = []
#     print('list neighbours',neighbours)
    for i in range(len(neighbours)):
        list_val.append(img[neighbours[i][1]][neighbours[i][2]][neighbours[i][3]])
    sum_var = np.std(list_val)
    return sum_var

    
def total_img_var(row, col, k, img):
    n, m, RGB = img.shape
    neighbours = find_neighbours(row, col, k, n, m, RGB)
    total_var = get_variation(img, neighbours)
    return total_var
    
    
def image_region_importance(img):
    # This function obtaines the image as an imput and it computes the importance of the 
    # different regions. 
    # 
    # Inputs:
    # - img: tensor of the image that we are considering
    # Outputs:
    # - probablity_matrix: matrix with the ecorresponding probabilities.
    
    n, m, k = img.shape
   
    probability_matrix = np.zeros((n, m, k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, img)
#                 print(probability_matrix[row,col,i])
    
    # We have to give a probability also to all the elelemnts that have zero variance
    # this implies that we will add to the whole matrix the minimum nonzero value, divided
    # by 100
    probability_matrix += np.min(probability_matrix[np.nonzero(probability_matrix)])/100
    
    # Normalise the probability matrix
    probability_matrix = probability_matrix/np.sum(probability_matrix)
    
    return probability_matrix

#########################################################
# Functions to subdivide into subregions the full pixels#
#########################################################


def nearest_of_the_list(i, j, k, n, m):
    """
    :param i: Row of the mxmx3 matrix in which we are
    :param j: Column of the mxmx3 Matrix in which we are
    :param k: Channel ...
    :param n: Dimension of the super-variable
    :param m: Dimension of the background matrix (n<m)
    :return: The relative elelemt of the super variable that is associated to i,j,k
    """
    x = np.linspace(0, 1, n)
    xx = np.linspace(0, 1, m)
    position_layer_x = np.argmin(np.abs(x-xx[j]))
    position_layer_y = np.argmin(np.abs(x-xx[i]))
    position_layer = position_layer_y*n + position_layer_x
    position_chann = k*n*n
    return position_layer + position_chann


def matr_subregions(var, n):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of beloging for each supervariable
    :param var: Image that we are perturbin. This will have shape (1,m,m,3)
    :param n: Dimension of the super grid that we are using
    :return: The matrix with the supervariable tto which each pixel belongs
    """
    # print(var.shape)
    A = var.copy()
    _, _, m, _ = A.shape
    for i in range(m):
        for j in range(m):
            for k in range(3):
                A[0, i, j, k] = nearest_of_the_list(i, j, k, n, m)
    return A


def matr_subregions_division(var, n):
    """
    This allows to compute the matrix fo dimension equal to the image with the
    region of belonging for each super-variable with non adjacent consideration
    :param var: Image that we are perturbing. This will have shape (1,m,m,3)
    :param n: Dimension of the super grid that we are using (n,n,3)
    :return: The matrix with the super-variable tto which each pixel belongs
    """
    # print(var.shape)
    A = var.copy().reshape(-1,)
    nn = n*n*3
    m = A.size
    mm = A.size

    lists = np.arange(nn)

    count = np.zeros((nn,))
    Tot = np.floor(m/nn)*np.ones((nn,))
    if np.mod(m, nn) != 0:
        Tot[range(np.mod(m, nn))] += 1

    # print('Check totality',np.sum(Tot)/m)
    prob = np.multiply(np.ones((nn, )), Tot)/m

    for i in range(m):
        # print(i, len(lists), np.sum(prob[lists]))
        cl = np.random.choice(len(lists), 1, replace=False, p=prob[lists])
        A[i] = lists[cl]
        mm -= 1
        count[lists[cl]] += 1
        if count[lists[cl]] == Tot[lists[cl]]:
            lists = np.delete(lists, cl)

        if mm != 0:
            prob = np.multiply(np.ones((nn,)), Tot-count)/mm

    return A.reshape(var.shape)


def matr_variance_subreg(var, n):
    """
        This allows to compute the matrix fo dimension equal to the image with the
        region of beloging for each supervariable with non adjacent consideration
        :param var: Image that we are perturbin. This will have shape (1,m,m,3)
        :param n: Dimension of the super grid that we are using (n,n,3)
        :return: The matrix with the supervariable tto which each pixel belongs
    """
    # print(var.shape)
    A = var.copy().reshape(-1, )
    nn = n*n*3
    mm = A.size
    n, m, k = var.shape
    # print(type(img))
    probability_matrix = np.zeros((n, m, k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, var)

    ordered = np.sort(probability_matrix.reshape(-1))[::-1]

    for i in range(nn):
        # We now cycle through the elements of the supervariable.
        idx = []
        nel = int(np.floor((mm - i)/nn))
        for j in range(nel):
            indices = np.where(probability_matrix.reshape(-1) == ordered[i + j*nn])[0]
            prev = ordered[:(i+j*nn)]
            prev_added = len(prev[prev == ordered[i+j*nn]])
            # print(ordered[i + j*nn], len(indices))
            idx.append(indices[prev_added])
        # print(idx)
        # print(i,' len',len(idx), ' pos ',idx[0])
        A[idx] = i
    #  checking that it actually works
    count = []
    for j in range(nn):
        count.append(len(A[A == j]))
    mean = np.mean(np.array(count))
    count_2 = 0
    for j in count:
        if np.abs(j-mean) > 2:
            count_2 += 1
    print(mean, count_2)
    return np.array([A.reshape(var.shape)])


def matr_power_subreg(var, n):
    """
        This allows to compute the matrix fo dimension equal to the image with the
        region of beloging for each supervariable with non adjacent consideration
        :param var: Image that we are perturbin. This will have shape (1,m,m,3)
        :param n: Dimension of the super grid that we are using (n,n,3)
        :return: The matrix with the supervariable tto which each pixel belongs
    """
    # print(var.shape)
    A = var.copy().reshape(-1, )
    nn = n*n*3
    mm = A.size
    n, m, k = var.shape
    # print(type(img))
    probability_matrix = np.zeros((n, m, k))

    for i in range(k):
        for row in range(n):
            for col in range(m):
                probability_matrix[row, col, i] = total_img_var(row, col, i, var)

    ordered = np.sort(probability_matrix.reshape(-1))[::-1]

    count = 0
    total = 0
    for i in range(nn+1):
        total += nn**2

    for i in range(nn):
        # We now cycle through the elements of the supervariable.
        idx = []
        nel = int(np.ceil(((i+1)**2)/total*mm))
        if count + nel >= mm:
            nel = mm - count
        # print('nel',nel, ' of ', mm, ' count', count)

        for j in range(nel):
            indices = np.where(probability_matrix.reshape(-1) == ordered[count + j])[0]
            prev = ordered[:(count + j)]
            prev_added = len(prev[prev == ordered[count + j]])
            # print(ordered[i + j*nn], len(indices))
            idx.append(indices[prev_added])
        # print(i,' len',len(idx))
        count += nel
        A[idx] = i
    return np.array([A.reshape(var.shape)])


def update_ortho_rand(dependency, small_x):
    n = small_x**2 * 3
    copy = dependency.copy()

    for i in range(n):
        idxs = np.where(dependency == i)
        # print(idxs)
        for j in range(len(idxs)):
            idx = (idxs[0][j], idxs[1][j], idxs[2][j], idxs[3][j])
            copy[idx] = np.mod(dependency[idx] + np.mod(j, n), len(idxs[0]))
    return copy

#########################################################
# SVR#
#########################################################


def Hier_Svd(x):
    n, m, l = x.shape
    temp = np.zeros((l, n, m))
    for i in range(n):
        for j in range(m):
            for k in range(l):
                temp[k, i, j] = x[i, j, k]

    u, s, vh = np.linalg.svd(temp)
    u2 = np.zeros((n, m, l))
    vh2 = np.zeros((n, m, l))

    for i in range(n):
        for j in range(m):
            for k in range(l):
                u2[i, j, k] = u[k, i, j]
                vh2[i, j, k] = vh[k, i, j]

    return temp, u2, s, vh2


def SVR_assignment(img, n1):
    n = n1*n1*3
    mm = img.size
    Assign = -1*np.ones(img.shape)
    Values = np.zeros(img.shape)
    # We first have to find the the different basis
    _, U, S, Vh = Hier_Svd(img)
    nn = len(S[0])*3
    # Generation of the comprehensive matrices

    n_cycle = np.minimum(n, nn)
    sing_values = np.sort(S.reshape(-1,))[::-1]
    for i in range(n_cycle):
        prec = sing_values[:i]
        dummy_s = len(prec[prec == sing_values[i]])
        pos = np.where(S == sing_values[i])
        num = pos[1][dummy_s]
        col = pos[0][dummy_s]
        temp = np.zeros(Values.shape)
        temp[:, :, col] = np.abs(U[:, :num, col]@np.diag(S[col][:num])@Vh[:num, :, col])
        # print(temp.size, Values.size)
        Assign[temp > Values] = i
        Values[temp > Values] = temp[temp > Values]
        # Values2[temp > Values] = S[col][num]
    # checking that all the pixels have been assigned
    if nn > n:
        temp = np.where(Assign < 0)
        if len(temp[0]) > 0:
            # randomly assign pixels that have to assignment
            for i in range(len(np.where(Assign < 0)[0])):
                Assign[temp[0][i], temp[1][i], temp[2][i]] = np.random.randint(n, size=1)
    else:
        A = Assign.reshape(-1,)
        temp = np.where(Assign < 0)
        if len(temp[0]) > 0:
            # randomly assign pixels that have to assignment
            for i in range(len(np.where(Assign < 0)[0])):
                Assign[temp[0][i], temp[1][i], temp[2][i]] = np.random.randint(n, size=1)
        # vectorise the pixels according to their singular value

        ordered = np.sort(Assign.reshape(-1))
        for i in range(n):
            # We now cycle through the elements of the supervariable.
            idx = []
            nel = int(np.floor((mm - i) / nn))
            for j in range(nel):
                indices = np.where(Assign.reshape(-1) == ordered[i + j * nn])[0]
                prev = ordered[:(i + j * nn)]
                prev_added = len(prev[prev == ordered[i + j * nn]])
                # print(ordered[i + j*nn], len(indices))
                idx.append(indices[prev_added])
            # print(i,' len',len(idx))
            A[idx] = i
        # checking that it actually works
        count = []
        for j in range(n):
            count.append(len(A[A == j]))
        mean = np.mean(np.array(count))
        count_2 = 0
        for j in count:
            if np.abs(j - mean) > 2:
                count_2 += 1
        if count_2 > 0:
            print('many unbalanced cases')
        Assign = A.reshape(Assign.shape)
        print('Assign has max', np.max(Assign), 'min', np.min(Assign), 'mean elements', mean, 'of which', count_2)
    print('Assign has max', np.max(Assign), 'min', np.min(Assign))
    return np.array([Assign], dtype=int)

#################################################################################################################
######################### TRYING TO DO SVD WITH SAME ELEMENTS FOR EACH SUPERVARIABLE ############################
#################################################################################################################


def pos_ordering_tensor(A, k=-1):
    """
    Returning the ordered list of elements in A and for each of them its location in A. If we want to select the
    highest value of A we will then select A[list(pos[0])] and so on
    :param A: Three dimensional tensor that we are trying to order
    :param k: number of elements we are interested in ordering. If =-1 then we will order the full tensor
    :return: ordered list and position list
    """
    if k < 0:
        k = A.size
    ordered = np.sort(A.reshape(-1,))[::-1]
    unique = np.sort(np.unique(ordered))[::-1]
    pos = np.array(np.where(A == unique[0])).T.reshape(-1, 3, 1)
    for i in range(1, k):
        pos = np.concatenate((pos, np.array(np.where(A == unique[i])).T.reshape(-1, 3, 1)))
    return ordered, pos


def pos_ordering_matrix(A, k=-1):
    """
    Returning the ordered list of elements in A and for each of them its location in A. If we want to select the
    highest value of A we will then select A[list(pos[0])] and so on
    :param A: Two dimensional tensor that we are trying to order
    :param k: number of elements we are interested in ordering. If =-1 then we will order the full tensor
    :return: ordered list and position list
    """
    if k < 0:
        k = A.size
    ordered = np.sort(A.reshape(-1,))[::-1]
    unique = np.sort(np.unique(ordered))[::-1]
    # print(unique)
    # print(unique[0])
    pos = np.array(np.where(A == unique[0])).T.reshape(-1, 2, 1)
    # print(pos)
    for i in range(1, k):
        pos = np.concatenate((pos, np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)))
    return ordered, pos


def pos_ordering_matrix_2(prec, A, k, Z):
    """
    Returning the ordered list of elements in A and for each of them its location in A. If we want to select the
    highest value of A we will then select A[list(pos[0])] and so on
    :param A: Two dimensional tensor that we are trying to order
    :param Z: Matrix with all the assignements. If -1 then element is not assigned.
    :return: ordered list and position list
    """
    ordered = np.sort(A.reshape(-1,))[::-1]
    unique = np.sort(np.unique(ordered))[::-1]
    count = 0
    i = prec
    dummy = True
    while dummy:
        # print(unique[i],i,prec,k)
        if np.max(Z[list(np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)[0])]) < 0:
            pos = np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)
            count += 1
            i += 1
            dummy = False
        else:
            # print('not finding')
            i += 1
    while count < k:
        # print(unique[i],i)
        if np.max(Z[list(np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)[0])]) < 0:
            pos = np.concatenate((pos, np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)))
            count += 1
            i += 1
        else:
            i += 1
    # print('end of pos',i, len(pos))
    return ordered, pos, i-prec


def pos_ordering_matrix_3(prec, A, k, Z):
    """
    Returning the ordered list of elements in A and for each of them its location in A. If we want to select the
    highest value of A we will then select A[list(pos[0])] and so on. Here we do not care about the element being
    already assigned.
    :param A: Two dimensional tensor that we are trying to order
    :param Z: Matrix with all the assignements. If -1 then element is not assigned.
    :return: ordered list and position list
    """
    ordered = np.sort(A.reshape(-1,))[::-1]
    unique = np.sort(np.unique(ordered))[::-1]
    count = 0
    i = prec
    dummy = True
    while dummy:
        pos = np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)
        count += 1
        i += 1
        dummy = False
    while count < k:
        # print(unique[i],i)
        # if np.max(Z[list(np.array(np.where(A == unique[i])).T.reshape(-1,2,1)[0])]) < 0:
        pos = np.concatenate((pos, np.array(np.where(A == unique[i])).T.reshape(-1, 2, 1)))
        count += 1
        i += 1

    # print('end of pos',i, len(pos))
    return ordered, pos, i-prec


def find_highest_elem(prec, d, M, A):
    orde, pos, skip = pos_ordering_matrix_2(prec, M, d, A)
    # skip = 0
    index = []
    for j in range(d):
        # print(list(pos[prec+j+skip]))
        while A[list(pos[j])] >= 0:
            print('had to skip')
            skip += 1
        index.append(list(pos[j]))
    done = prec+skip+d
    return np.array(index), done


def expand_index(index, col):
    l = len(index)
    temp = []
    for j in range(l):
        temp.append(np.append(index[j], col))
    return np.array(temp).reshape(l, 3, 1)

#########################################################################################################
#################################   SIGN CASE
#########################################################################################################


def find_highest_elem_2(prec, d, M, A):
    orde, pos, skip = pos_ordering_matrix_3(prec, M, d, A)
    # skip = 0
    signs = []
    index = []
    for j in range(d):
        # print(list(pos[prec+j+skip]))
        while A[list(pos[j])] >= 0:
            print('had to skip')
            skip += 1
        index.append(list(pos[j]))
        signs.append(np.sign(M[list(pos[j])]))
    done = prec+skip+d
    return np.array(index), done, signs


def find_highest_elem_3(prec, d, M, A):
    """
    This keeps does not keep track of the matrix A being already asigned
    :param prec:
    :param d:
    :param M:
    :param A:
    :return:
    """
    orde, pos, skip = pos_ordering_matrix_3(prec, M, d, A)
    # skip = 0
    signs = []
    index = []
    for j in range(d):
        # print(list(pos[prec+j+skip]))
        # while A[list(pos[j])] >= 0:
        #     print('had to skip')
        #     skip += 1
        index.append(list(pos[j]))
        signs.append(np.sign(M[list(pos[j])]))
    done = prec+skip+d
    return np.array(index), done, signs


def assign_index(Assig, index, el):
    # We assign index to pos of Assign. If this is 0 we will then increase
    # the dimension of the Matrix Assign
    Assign = Assig.copy()
    dummy = True

    # try to associate the value to the nearest dimension
    j = 0
    n = Assign.shape[-1]
    # print('Inside assign n is', n, 'and assign has shape', Assign.shape)
    # print('Assign at index is', Assign[index], 'with shape', Assign[index].shape)
    while dummy and j < n:
        index_t = index.copy()
        index_t.append(j)
        if Assign[index_t] == -1:
            Assign[index_t] = el
            # print('Just assignes ', Assign[index][j], 'instead of ', el)
            dummy = False
            # print('value assigned')
        j += 1

    # we have to increase dimension if not still assigned the index
    if dummy:
        temp = list(Assign.shape[:-1])
        temp.append(1)
        Increase = -1*np.ones(tuple(temp))
        Assign = np.concatenate((Assign, Increase), axis=len(temp) - 1)
        index_t = index.copy()
        index_t.append(j)
        Assign[index_t] = el
        # print('We got inside the increase dimension')

    # print('Assigned value', Assign[index][j-1], ' whole vector ', Assign[index])
    return Assign


def finding_indices(dependency, index):
    # This returns a matrix with the elements that are equal to index
    # print(np.array(dependency == index).shape)
    return np.any(dependency == index, axis=len(dependency.shape)-1)


def SVR_assignment_equal_2(img, n1):
    n = n1*n1*3
    mm = int(img.size*0.5)  # /2?
    shape1 = list(img.shape)
    shape1.append(1)
    shape = tuple(shape1)
    Assign = -1*np.ones(shape)
    Values = np.ones(img.shape)
    # We first have to find the the different basis
    _, U, S, Vh = Hier_Svd(img)
    nn = int((len(S[0][S[0] > 0]) + len(S[1][S[1] > 0]) + len(S[2][S[2] > 0]))*0.5)

    # Generate an order of the highest singular values in all the three dimensions
    sing_values, pos = pos_ordering_matrix(S)
    # capacities = assign_capacities(U,S,Vh,nn)
    # Generation of the comprehensive matrices
    d = int(np.floor(mm/n))
    remaining = int(np.mod(mm, n))
    # print(remaining)
    max_m = int(np.ceil(mm/nn))
    print('Total of ', mm, ' pixels devided between ', nn, ' super variables; ', d, 'each')
    prec = 0
    done = 0
    assigned = 0
    c = 0
    # pos = np.array(np.where(S == sing_values[c])).T.reshape(-1,2)
    # print('Pos',pos[0])
    # prec_val = sing_values[:c+1]
    # print('prec Val',prec_val)
    # dummy_s = len(prec_val[prec_val == sing_values[c]])
    # num = int(pos[c][1])
    # col = int(pos[c][0])
    done_channel = np.zeros((3,))
    # print('indices',num,col)
    # M = np.abs(U[:, :num+1, col] @ np.diag(S[col][:num+1]) @ Vh[:num+1, :, col])
    # print('M is',np.max(M), np.min(M), 'and S is ',S[col][:num+1])
    # print('Have ',n,' dimensional supervector whose every single variable is related to ', d, ' pixels')
    # print('There are a maximum of ',max_m, 'elements takend per SVD')
    c = -1
    rem = mm/3
    bar = progressbar.ProgressBar(maxval=n,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(n):
        # print('Supervariable',i/n, end='',flush=True)
        bar.update(i + 1)

        while assigned < d:

            if prec == 0:
                c += 1
                num = int(pos[c][1])
                col = int(pos[c][0])
                rem = mm / 3 - done_channel[col]
                # print('Sing Value number ',c,' in channel ', col,' with remainig', rem,' to do ',d)
                M = S[col][num] * np.abs(np.outer(U[:, num, col], Vh[num, :, col]))

            not_assigned = d-assigned
            space_in_the_Svd = max_m - prec
            to_assign = int(np.minimum(not_assigned,
                                   space_in_the_Svd))
            rem = mm/3-done_channel[col]
            to_assign = int(np.minimum(to_assign, rem))
            # print('not asssigned', not_assigned, ' with space ', space_in_the_Svd)
            # print('to_assign',to_assign)
            # print(Assign[:,:,col,:].shape)
            index, done, signs = find_highest_elem_3(done, to_assign, M, Assign[:, :, col, :])
            # print('index', list(index[0]))
            index = expand_index(index, col)
            assigned += to_assign
            done_channel[col] += to_assign
            for k in range(len(index)):
                # print('index',list(index[k]))#,'Assign',Assign[list(index[k])])
                Assign = assign_index(Assign, list(index[k]), i)
                # Values[list(index[k])] = signs[k]

            if to_assign + prec == max_m:
                # print(len(Assign[Assign>=0]),' with ',assigned, 'of',d,' when done ', done)
                done = 0
                prec = 0
            elif to_assign == rem:
                done = 0
                prec = 0
            else:
                # print('upper',done,prec)
                prec += to_assign
                # print('prec',prec)
        assigned = 0
    bar.finish()
    print(Assign.shape)
    count = []
    for j in range(n):
        count.append(len(Assign[Assign == j]))
    mean = np.mean(np.array(count))
    count_2 = 0
    for j in count:
        if np.abs(j - mean) > 2:
            count_2 += 1
    if count_2 > 0:
        print('many unbalanced cases')
    # Assign = Assign.reshape(Assign.shape)
    # print('Assign has max', np.max(Assign), 'min', np.min(Assign), 'mean elements', mean, 'of which', count_2)
    return np.array([Assign], dtype=int), np.array([Values])


class BlackBox_BOBYQA:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, max_iterations=MAX_ITERATIONS,
                 print_every=100, early_stop_iters=0,
                 abort_early=ABORT_EARLY,
                 use_log=False, use_tanh=True, use_resize=False,
                 start_iter=0, L_inf=0.15,
                 init_size=5, use_importance=True, rank=2,
                 ordered_domain=False, image_distribution=False,
                 mixed_distributions=False, Sampling=True, Rank=False,
                 Rand_Matr=False, Rank_1_Mixed =False, GenAttack=False,
                 max_eval=1e5):
        """
        The BOBYQA attack.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        """
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.model = model
        self.sess = sess
        self.rank = rank
        self.TARGETED = targeted
        self.target = 0
        self.Generations = int(1e5)
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        self.use_importance = use_importance
        self.ordered_domain = ordered_domain
        self.image_distribution = image_distribution
        self.mixed_distributions = mixed_distributions
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size
        
        self.L_inf = L_inf
        self.use_tanh = use_tanh
        self.use_resize = use_resize

        self.max_eval = max_eval

        self.Sampling = Sampling
        self.Rank = Rank
        self.Rand_Matr = Rand_Matr
        
        # each batch has a different modifier value (see below) to evaluate
        single_shape = (image_size, image_size, num_channels)
        small_single_shape = (self.small_x, self.small_y, num_channels)

        # the variable we're going to optimize over
        # support multiple batches
        # support any size image, will be resized to model native size
        if self.use_resize:
            self.modifier = tf.placeholder(tf.float32, shape=(None, None, None, None))
            # scaled up image
            self.scaled_modifier = tf.image.resize_images(self.modifier, [image_size, image_size],
                                                          method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            # operator used for resizing image
            self.resize_size_x = tf.placeholder(tf.int32)
            self.resize_size_y = tf.placeholder(tf.int32)
            self.resize_input = tf.placeholder(tf.float32, shape=(1, None, None, None))
            self.resize_op = tf.image.resize_images(self.resize_input, [self.resize_size_x, self.resize_size_y],
                                                    align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
        else:
            self.modifier = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            # no resize
            self.scaled_modifier = self.modifier
        # the real variable, initialized to 0
        self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        
        # these are variables to be more efficient in sending data to tf
        # we only work on 1 image at once; the batch is for evaluation loss at different modifiers
        self.timg = tf.Variable(np.zeros(single_shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)
        
        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, single_shape)
        self.assign_tlab = tf.placeholder(tf.float32, num_labels)
        
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # broadcast self.timg to every dimension of modifier
        if use_tanh:
            self.newimg = tf.tanh(self.scaled_modifier + self.timg)/2
        else:
            self.newimg = self.scaled_modifier + self.timg

        # prediction BEFORE-SOFTMAX of the model
        # now we have output at #batch_size different modifiers
        # the output should have shape (batch_size, num_labels)
        self.output = model.predict(self.newimg)
        # compute the probability of the label class versus the maximum other
        # self.tlab * self.output selects the Z value of real class
        # because self.tlab is an one-hot vector
        # the reduce_sum removes extra zeros, now get a vector of size #batch_size
        self.real = tf.reduce_sum(self.tlab*self.output, 1)
        
        # (1-self.tlab)*self.output gets all Z values for other classes
        # Because soft Z values are negative, it is possible that all Z values are less than 0
        # and we mistakenly select the real class as the max. So we minus 10000 for real class
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000), 1)
        self.use_max = False
        self.sum = tf.reduce_sum((1-self.tlab)*self.output, 1)
        # self.logsum  = tf.reduce_sum(tf.log((1-self.tlab)*self.output- (self.tlab*10000)),1)
        # _,index_ord  = tf.nn.top_k(self.tlab*self.output,10)
        # self.temp    = tf.gather(self.output,tf.cast(index_ord[0],tf.int32))
        # self.k_max   = tf.reduce_sum(self.temp,0)
        # If self.targeted is true, then the targets represents the target labels.
        # If self.targeted is false, then targets are the original class labels.
        if self.TARGETED:
            # The loss is log(1 + other/real) if use log is true, max(other - real) otherwise
            self.loss = tf.log(tf.divide(self.sum + 1e-30, self.real+1e-30))
            # self.loss_max = tf.log(tf.divide(self.other +1e-30,self.real+1e-30))
            self.distance = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.used_var_list = np.zeros(var_size, dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

        # upper and lower bounds for the modifier
        self.image_size = image_size
        self.num_channels = num_channels
        self.var_size_b = image_size * image_size * num_channels
        self.modifier_up = np.zeros(self.var_size_b, dtype=np.float32)
        self.modifier_down = np.zeros(self.var_size_b, dtype=np.float32)
        
    def resize_img(self, small_x, small_y, reset_only=False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.num_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(self.real_modifier)
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x,
                                                                          self.resize_size_y: self.small_y,
                                                                          self.resize_input: self.real_modifier})
            print(self.real_modifier.shape)
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)

    def blackbox_optimizer(self, iteration):
        # build new inputs, based on current variable value
        var = self.real_modifier.copy()
        var_size = self.real_modifier.size
        if self.use_importance:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False, p=self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.size, self.batch_size, replace=False)
        indice = self.var_list[var_indice]        
        opt_fun = lambda c: self.sess.run([self.loss], feed_dict={self.modifier: vec2mod(c, indice, var)})[0]
        
        x_o = np.zeros(self.batch_size,)
        b = self.modifier_up[indice]
        a = self.modifier_down[indice]
        
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg=self.L_inf/3,
                              bounds=(a, b), maxfun=self.batch_size*1.1,
                              npt=self.batch_size+1)
        evaluations = soln.nf

        # adjust sample probability, sample around the points with large gradient
        nimgs = vec2mod(soln.x, indice, var)
        
        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)

        return soln.f, evaluations, nimgs        
        
    def blackbox_optimizer_ordered_domain(self, iteration, ord_domain, Random_Matrix, super_dependency, img):
        # build new inputs, based on current variable value
        times = np.zeros(8,)
        times[0] = time.time()
        var = 0*np.array([img])  # self.real_modifier.copy()
        # print('the type of var is', type(var[0]))
        # print('the shape of var is',var[0].shape)
        var_size = self.real_modifier.size
        # print(np.max(super_dependency))
        NN = self.var_list.size
        nn = self.batch_size

        # We choose the elements of ord_domain that are inherent to the step. So it is already
        # limited to the variable's dimension
        if (iteration+1)*nn <= NN:
            var_indice = ord_domain[iteration*nn: (iteration+1)*nn]
        else:
            var_indice = ord_domain[list(range(iteration*nn, NN)) + list(range(0, (self.batch_size-(NN-iteration*nn))))]
            
        indice = self.var_list[var_indice]
        # print(indice)
        # Changing the bounds according to the problem being resized or not
        if self.use_resize:
            # print('super_def',super_dependency.shape)
            # print('modifier',self.modifier_down.shape)
            a = np.zeros((nn,))
            b = np.zeros((nn,))

            for i in range(nn):
                # print(len(self.modifier_down[super_dependency.reshape(-1,) == indice[i]])/len(super_dependency.reshape
                # count_t += len(self.modifier_down[super_dependency.reshape(-1,) == indice[i]])/len(super_dependency.r
                indices = finding_indices(super_dependency, indice[i]).reshape(-1,)
                # print(indices)
                # super_dependency.reshape(-1,) == indice[i]
                up = np.maximum(self.modifier_up[indices]*Random_Matrix.reshape(-1,)[indices],
                                self.modifier_down[indices]*Random_Matrix.reshape(-1,)[indices])
                down = np.minimum(self.modifier_down[indices]*Random_Matrix.reshape(-1,)[indices],
                                  self.modifier_up[indices]*Random_Matrix.reshape(-1,)[indices])
                a[i] = np.min(down)
                b[i] = np.max(up)
            # print('total is ', count_t)
            # print(b)
            # print('The constraint interval is', np.mean(np.abs((b-a))))
        else:
            b = self.modifier_up[indice]
            a = self.modifier_down[indice]
        bb = self.modifier_up
        aa = self.modifier_down
        # times[1] = time.time()
        opt_fun = Objfun(lambda c: self.sess.run([self.loss], feed_dict={
            self.modifier: vec2modMatRand2(c, indice, var, Random_Matrix, super_dependency, bb, aa)})[0])
        x_o = np.zeros(self.batch_size,)
        times[1] = time.time()
        times[2] = time.time()
        # Checking if the function evaluation changs real_modifier
        # if not (var - self.real_modifier).any() == 0:
        #     print('Not the same after the evalluation')
        #     print(var.shape,type(var))
        #     print(self.real_modifier,type(self.real_modifier))

        times[3] = time.time()
        initial_loss = opt_fun(x_o)
        # print(initial_loss)
        times[4] = time.time()
        times[5] = time.time()
        soln = pybobyqa.solve(opt_fun, x_o, rhobeg=np.min(b-a)/3,
                              bounds=(a, b), maxfun=self.batch_size*1.1,
                              rhoend=np.min(b-a)/6,
                              npt=self.batch_size+1)
        times[6] = time.time()
        summary = opt_fun.get_summary(with_xs=False)
        # TO DO: get the difference vaLue
        evaluations = soln.nf
        # adjust sample probability, sample around the points with large gradient
        # print('In the superdependency there are ', len(super_dependency[super_dependency>-1]),'elements assigned')
        # print(soln)
        nimgs = vec2modMatRand2(soln.x, indice, var, Random_Matrix, super_dependency, bb, aa)
        distance = self.sess.run(self.distance, feed_dict={self.modifier: nimgs})
        # print('Nonzero elements',np.count_nonzero(nimgs)/nimgs.size)
        if soln.f > initial_loss:
            # print('The optimisation is not working. THe diff is ', initial_loss - soln.f)
            return initial_loss, evaluations + 1, self.real_modifier, times, summary
        else:
            # print('The optimisation is working. THe diff is ', initial_loss - soln.f)
            times[7] = time.time()
            return distance[0], evaluations + 1, nimgs, times, summary

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        self.target = np.argmax(targets)
        print('The target is', self.target)
        print('go up to', len(imgs))
        # we can only run 1 image at a time, minibatches are used for gradient evaluation
        for i in range(0, len(imgs)):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i], targets[i]))
        return np.array(r)

    def attack_batch(self, img, lab):
        """
        Run the attack on a batch of images and labels.
        """
        self.target = np.argmax(lab)
        
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # remove the extra batch dimension
        
        if self.mixed_distributions & self.image_distribution:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Sampling & self.Rank & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Sampling & self.Rank:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Sampling & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        if self.Rank & self.Rand_Matr:
            print('We cannot acept both mixed_distribution and image_distribution to be True')
            return -1
        
        if len(img.shape) == 4:
            img = img[0]
        if len(lab.shape) == 2:
            lab = lab[0]
        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        upper_bound = 1e10

        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)
        img0 = img
        
        # clear the modifier
        resize_iter = 0
        if self.use_resize:
            self.resize_img(self.resize_init_size, self.resize_init_size, True)
        else:
            self.real_modifier.fill(0.0)

        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = img
        eval_costs = 0
        
        if self.ordered_domain:
            print(np.random.choice(10, 3))
            ord_domain = np.random.choice(self.var_list.size, self.var_list.size, replace=False, p=self.sample_prob)
            
        started_with_log_now_normal = False
        
        iteration_scale = -1

        self.sess.run(self.setup, {self.assign_timg: img,
                                   self.assign_tlab: lab})

        previous_loss = 1e6
        count_steady = 0
        global_summary = []
        first_renew = True
        adv = 0 * img
        for step in range(self.MAX_ITERATIONS):
            # use the model left by last constant change
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0

            attack_time_step = time.time()

            # print out the losses every 10%
            if step % self.print_every == 0:
                loss, output, distance = self.sess.run((self.loss, self.output, self.distance),
                                                       feed_dict={self.modifier: np.array([adv])})
                print("[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss_f = {:.5g}, maxel = {}".format(step,
                                                eval_costs, train_timer, self.real_modifier.shape, distance[0], loss[0], np.argmax(output[0])))
                sys.stdout.flush()

            attack_begin_time = time.time()
            printing_time_initial_info = attack_begin_time - attack_time_step
            
            zz = np.zeros(img.shape)
            ub = 0.5*np.ones(img.shape)
            lb = -ub
            
            if self.use_resize:
                zz = np.zeros((self.image_size, self.image_size, self.num_channels))
                ub = 0.5*np.ones((self.image_size, self.image_size, self.num_channels))
                lb = -ub
            
            if not self.use_tanh:  # and (step>0):
                self.modifier_up = np.maximum(np.minimum(- (img.reshape(-1,) - img0.reshape(-1,)) + self.L_inf,
                                                         ub.reshape(-1,) - img.reshape(-1,)),
                                              zz.reshape(-1,))
                self.modifier_down = np.minimum(np.maximum(- (img.reshape(-1,) - img0.reshape(-1,)) - self.L_inf,
                                                           - img.reshape(-1,) + lb.reshape(-1,)),
                                                zz.reshape(-1,))
            
            if step > 0:
                self.sess.run(self.setup, {self.assign_timg: img,
                                           self.assign_tlab: lab})
                self.real_modifier.fill(0.0)
            
            if self.ordered_domain:
                iteration_scale += 1
                iteration_domain = np.mod(iteration_scale, (self.use_var_len//self.batch_size + 1))

                # print(iteration_domain,iteration_scale)
                force_renew = False
                # print('iteration_domain ', iteration_domain)
                if iteration_domain == 0:
                    # We force to regenerate a distribution of the nodes if we have 
                    # already optimised over all of the domain
                    force_renew = True

                steps = [4, 16, 25, 40, 55, 75, 100, 125]
                dimen = [8, 12, 20, 30, 45, 70, 100,  40]

                if self.use_resize:

                    if step in steps:
                        idx = steps.index(step)
                        self.small_x = dimen[idx]
                        self.small_y = dimen[idx]
                        self.resize_img(self.small_y, self.small_x, False)
                        iteration_scale = 0
                        iteration_domain = 0
                        force_renew = True
                        first_renew = True

                if (np.mod(iteration_scale, 50) == 0) or force_renew:
                    # We have to restrt the random matrix and the

                    if first_renew:
                        super_dependency, Random_Matrix = SVR_assignment_equal_2(img, self.small_x)
                        print('In the superdependency there are ', len(super_dependency[super_dependency > -1]),
                              'elements assigned')
                        # Random_Matrix = np.ones(np.array([img]).shape)
                        first_renew = False

                    # Random_Matrix =  np.ones(np.array([img]).shape)#np.random.random_sample(np.array([img]).shape)
                    print('Regeneration')

                    # We have to repermute the pixels if they are modifie

                    if self.image_distribution:

                        if self.use_resize:
                            prob = image_region_importance(tf.image.resize_images(img, [self.small_x, self.small_y],
                                                                                  align_corners=True,
                                                                                  method=
                                                                                  tf.image.ResizeMethod.BILINEAR).eval()
                                                           ).reshape(-1,)
                        else:
                            prob = image_region_importance(img).reshape(-1,)

                        ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p=prob)
                    elif self.mixed_distributions:
                        if step == 0:
                            prob = image_region_importance(img).reshape(-1,)
                            ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False, p=prob)
                        else:
                            ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False,
                                                          p=self.sample_prob)
                    else:
                        ord_domain = np.random.choice(self.use_var_len, self.use_var_len, replace=False,
                                                      p=self.sample_prob)
                time_before_calling_attack_batch = time.time()

                l, evaluations, nimg, times, summary = self.blackbox_optimizer_ordered_domain(iteration_domain,
                                                                                              ord_domain,
                                                                                              Random_Matrix,
                                                                                              super_dependency,
                                                                                              img)
            else:
                # Normal perturbation method
                l, evaluations, nimg = self.blackbox_optimizer(step)

            global_summary.append(summary)
            
            time_after_calling_attack_batch = time.time()

            if (previous_loss - l) < 1e-3:
                count_steady += 1
            previous_loss = l
            
            adv = nimg[0]
            
            temp = np.minimum(self.modifier_up.reshape(-1,),
                              adv.reshape(-1,))
            adv = temp
            adv = np.maximum(self.modifier_down,
                             adv)
            adv = adv.reshape((self.image_size, self.image_size, self.num_channels))
            img = img + adv
            eval_costs += evaluations

            time_modifying_image = time.time()

            # check if we should abort search if we're getting nowhere.
            if self.ABORT_EARLY and step % self.early_stop_iters == 0:
                if l > prev*.9999:
                    print("Early stopping because there is no improvement")
                    return o_bestattack, eval_costs
                prev = l

            # Find the score output
            loss_test, score, real, other = self.sess.run((self.loss, self.output, self.real, self.other),
                                                          feed_dict={self.modifier: np.array([adv])})
                
            score = score[0]
                        
            # adjust the best result found so far
            # the best attack should have the target class with the largest value,
            # and has smallest l2 distance
            
            if l < o_bestl2 and compare(score, np.argmax(lab)):
                # print a message if it is the first attack found
                if o_bestl2 == 1e10:
                    print("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}".format(
                        step, eval_costs, train_timer, self.real_modifier.shape, l))
                    sys.stdout.flush()
                o_bestl2 = l
                o_bestscore = np.argmax(score)
                o_bestattack = img 
                
            # If loss if 0 return the result
            if eval_costs > self.max_eval:
                print('The algorithm did not converge')
                return o_bestattack, eval_costs, global_summary

            if l <= 0:
                print("Early Stopping becuase minimum reached")
                return o_bestattack, eval_costs, global_summary

            time_checking_conv = time.time()

            train_timer += time.time() - attack_begin_time

            # print('Total time of the iteration:', time_checking_conv - attack_time_step)
            # # print('-- initial info:',printing_time_initial_info)
            # # print('-- faff before batch:', time_before_calling_attack_batch - attack_begin_time)
            # print('-- batch attack:', time_after_calling_attack_batch-time_before_calling_attack_batch)
            # # print('-- -- initialisation', times[1]-times[0])
            # print('-- -- 2vec', times[2] - times[1])
            # # print('-- -- check', times[3] - times[2])
            # print('-- -- fun_eval', times[4] - times[3])
            # # print('-- -- boundaries', times[5] - times[4])
            # print('-- -- BOBYQA', times[6] - times[5])
            # print('-- -- modifier', times[7] - times[6])
            # print('-- modifying time:', time_modifying_image-time_after_calling_attack_batch)
            # print('-- check converg:', time_checking_conv-time_modifying_image)
        # return the best solution found
        return o_bestattack, eval_costs, global_summary
