# import cv2
import itertools
import math
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import sys
import time

from Attack_Code.Combinatorial.attacks.local_search_helper_function_adv_inception import LocalSearchHelper_adv_inception


def ParsimoniousAttack_function_adv_inception(loss_f, image, values,max_internal_queries, noise,
                                _perturb_image,label, args, **kwargs):
    """Parsimonious attack using local search algorithm"""

    # Hyperparameter setting
    loss_func = args.loss_func
    max_queries = args.max_queries
    epsilon = args.epsilon
    batch_size = args.batch_size
    no_hier = args.no_hier
    block_size = args.block_size
    no_hier = args.no_hier
    num_channels = args.num_channels

    def _split_block(upper_left, lower_right, block_size, channels):
        """Split an image into a set of blocks.
        Note that a block consists of [upper_left, lower_right, channel]

        Args:
            upper_left: [x, y], the coordinate of the upper left of an image
            lower_right: [x, y], the coordinate of the lower right of an image
            block_size: int, the size of a block

        Returns:
            blocks: list, the set of blocks
        """
        blocks = []
        xs = np.arange(upper_left[0], lower_right[0], block_size)
        ys = np.arange(upper_left[1], lower_right[1], block_size)
        for x, y in itertools.product(xs, ys):
            for c in range(channels):
                blocks.append([[x, y], [x+block_size, y+block_size], c])
        return blocks
  
    # Class variables
    width = image.shape[0]
    height = image.shape[1]
    channels = image.shape[2]
    adv_image = np.copy(image)
    num_queries = 0

    if values is not None:
        block_size = values['block_size']
        num_queries = values['num_queries']

    upper_left = [0, 0]
    if no_hier:
        lower_right = [width,height]
        block_size = 1
    else:
        lower_right = [256, 256]

    # Split an image into a set of blocks
    blocks = _split_block(upper_left, lower_right, block_size, num_channels)

    # Initialize a noise to -epsilon
    if noise is None:
        if not no_hier:
            noise = -epsilon*np.ones([1, 256, 256, channels], dtype=np.float32)
        else:
            noise = -epsilon*np.ones([1, width, height, channels], dtype=np.float32)


    # Construct a batch
    num_blocks = len(blocks)
    batch_size = batch_size if batch_size > 0 else num_blocks
    if values is None:
        curr_order = np.random.permutation(num_blocks)
        loss = np.inf
    else: 
        curr_order = values['curr_order']
        loss = values['loss']
    
    time_beg = time.time()
    time_end = time.time()
    if values is not None:
        initial_batch = values['batch']+1
    else:
        initial_batch = 0

    internal_queries = 0

    # Main loop
    while True:
        # Run batch
        num_batches = int(math.ceil(num_blocks/batch_size))
        for i in range(initial_batch, num_batches):
            try:
                print("[STATS][L2] rate = {:.5g}, cost = {}, size = {}, loss = {:.5g}, time = {:.5g}".
                        format(i/num_batches, num_queries, block_size, loss, time_end-time_beg))
                sys.stdout.flush()
            except:
                print('could not print', loss)
            time_beg = time.time()
            # Pick a mini-batch
            bstart = i*batch_size
            bend = min(bstart + batch_size, num_blocks)
            blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
            # Run local search algorithm on the mini-batch

            
            noise, queries, loss, success = LocalSearchHelper_adv_inception(loss_f, image, noise,
                                blocks_batch, num_queries,
                                _perturb_image, label, args)
            time_end = time.time()
            num_queries += queries
            internal_queries += queries
            
            # setting values to transmit
            values = {'batch':i, 'block_size':block_size, 'curr_order':curr_order,
                        'num_queries': num_queries, 'loss':loss}

            # Generate an adversarial image
            # print('Going for the perturbation')
            adv_image = _perturb_image(width, height, image, noise)
            # If query count exceeds the maximum queries, then return False
            if internal_queries > max_internal_queries:
                return adv_image, num_queries, False, values, noise
            
            if num_queries > max_queries:
                return adv_image, num_queries, True, values, noise
            # If attack succeeds, return True
            if success:
                return adv_image, num_queries, True, values, noise

        # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
        if not no_hier and block_size >= 2:
            block_size //= 2
            blocks = _split_block(upper_left, lower_right, block_size, num_channels)
            num_blocks = len(blocks)
            batch_size = batch_size if batch_size > 0 else num_blocks
            curr_order = np.random.permutation(num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
            curr_order = np.random.permutation(num_blocks)
