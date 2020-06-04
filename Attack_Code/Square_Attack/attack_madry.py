import os
import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import random
import time

import pickle

from PIL import Image


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def square_attack_linf(model, x, y, eps, n_iters, p_init, targeted, loss_type, print_every=50):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = -0.5, 0.5 if x.max() <= 1 else 255
    h, w, c = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], 1, w, c])
    x_best = np.clip(x, min_val, max_val)

    logits = model.predict(x_best)
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    print('Initial Loss = ', loss_min)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr, y_curr = x, x_best, y
        loss_min_curr, margin_min_curr = loss_min, margin_min
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            x_best_curr_window = x_best_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, center_h:center_h+s, center_w:center_w+s, :], 
                                        min_val, max_val) 
                                - x_best_curr_window) 
                        < 10**-7) == c*s*s:
                deltas[i_img, center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[1, 1, c])
        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min = (idx_improved) * loss + (~idx_improved) * loss_min_curr
        margin_min = (idx_improved) * margin + (~idx_improved) * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best = (idx_improved) * x_new + (~idx_improved) * x_best_curr
        n_queries += 1
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start

        if np.mod(i_iter, print_every)==0:
            print('[L1] {}: margin={}, loss ={:.3f}, p={}  (n_ex={}, eps={:.3f}, {:.2f}s)'.
                format(i_iter+1, margin_min[0], loss_min[0], p, x.shape[0], eps, time_total))

        if acc == 0:
            break
    
    if len(x_best.shape) == 3:
            x_best = x_best.reshape((1,) + x_best.shape)
    return x_best , n_queries, [], acc==0
