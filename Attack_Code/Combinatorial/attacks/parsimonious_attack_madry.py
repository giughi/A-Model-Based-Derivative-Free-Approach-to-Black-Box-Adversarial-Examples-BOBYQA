# import cv2
import itertools
import math
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import sys
import time
import cv2 

from Attack_Code.Combinatorial.attacks.local_search_helper_madry import LocalSearchHelper


def ParsimoniousAttack(loss_f, image, label, args, **kwargs):
    """Parsimonious attack using local search algorithm"""

    # Hyperparameter setting
    max_queries = args.max_evals
    epsilon = args.eps
    batch_size = args.batch_size
    no_hier = args.no_hier
    block_size = args.block_size
    no_hier = args.no_hier
    num_channels = args.num_channels
    dim_image = args.dim_image

    def _perturb_image(width, height, image, noise):
            """Given an image and a noise, generate a perturbed image.
            First, resize the noise with the size of the image.
            Then, add the resized noise to the image.

            Args:
                image: numpy array of size [1, 299, 299, 3], an original image
                noise: numpy array of size [1, 256, 256, 3], a noise

            Returns:
                adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
            """
            adv_image = image + cv2.resize(noise[0, ...], (width, height), interpolation=cv2.INTER_NEAREST)
            if width != 96:
                adv_image = np.clip(adv_image, -0.5, 0.5)
            else:
                adv_image = np.clip(adv_image, -1, 1)
            return np.array([adv_image])


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

    upper_left = [0, 0]
    if no_hier:
        lower_right = [width,height]
        block_size = 1
    else:
        lower_right = [256, 256]

    # Split an image into a set of blocks
    blocks = _split_block(upper_left, lower_right, block_size, num_channels)

    # Initialize a noise to -epsilon
    if not no_hier:
        noise = -epsilon*np.ones([1, 256, 256, channels], dtype=np.float32)
    else:
        noise = -epsilon*np.ones([1, width, height, channels], dtype=np.float32)


    # Construct a batch
    num_blocks = len(blocks)
    batch_size = batch_size if batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)
    loss = np.inf

    
    time_beg = time.time()
    time_end = time.time()
    initial_batch = 0

    internal_queries = 0

    # Main loop
    while True:
        # Run batch
        num_batches = int(math.ceil(num_blocks/batch_size))
        # print('We got ', num_blocks,' blocks and use batches of dimension ', batch_size)
        for i in range(initial_batch, num_batches):
            # print(i, num_batches)
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

            
            noise, queries, loss, success = LocalSearchHelper(loss_f, image, noise,
                                blocks_batch, num_queries, label, args)
            time_end = time.time()
            num_queries += queries
            internal_queries += queries
            
            # Generate an adversarial image
            # print('Going for the perturbation')
            adv_image = _perturb_image(width, height, image, noise)
            # If query count exceeds the maximum queries, then return False
            
            if num_queries > max_queries:
                return adv_image[0], num_queries, [], success
            # If attack succeeds, return True
            if success:
                return adv_image[0], num_queries, [], success

        # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
        
        if not no_hier and block_size >= 2 and block_size/256*dim_image>1:
            # print('CHANGING BLOCK SIZE')
            block_size //= 2
            blocks = _split_block(upper_left, lower_right, block_size, num_channels)
            num_blocks = len(blocks)
            batch_size = batch_size if batch_size > 0 else num_blocks
            curr_order = np.random.permutation(num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
            curr_order = np.random.permutation(num_blocks)
