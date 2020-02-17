# import cv2
import itertools
import math
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import sys
import time

from Attack_Code.Combinatorial.attacks.local_search_helper_3 import LocalSearchHelper_2


class ParsimoniousAttack_restarts(object):
    """Parsimonious attack using local search algorithm"""

    def __init__(self, model, args, **kwargs):
        """Initialize attack.

        Args:
            model: TensorFlow model
            args: arguments
        """
        # Hyperparameter setting
        self.loss_func = args.loss_func
        self.max_queries = args.max_queries
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.no_hier = args.no_hier
        self.block_size = args.block_size
        self.no_hier = args.no_hier
        self.max_iters = args.max_iters
        self.model = model
        self.num_channels = args.num_channels

        # Create helper
        self.local_search = LocalSearchHelper_2(model, args)
 
    def _perturb_image(self, image, noise):
        """Given an image and a noise, generate a perturbed image.
        First, resize the noise with the size of the image.
        Then, add the resized noise to the image.

        Args:
            image: numpy array of size [1, 299, 299, 3], an original image
            noise: numpy array of size [1, 256, 256, 3], a noise

        Returns:
            adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
        """
        # adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        resized_img = tf.image.resize_images(np.array([noise[0,...]]), [self.width, self.height],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True).eval()
        adv_image = image + resized_img
        if self.width != 96:
          adv_image = np.clip(adv_image, -0.5, 0.5)
        else:
          adv_image = np.clip(adv_image, -1, 1)
        return adv_image

    def _split_block(self, upper_left, lower_right, block_size, channels):
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
  
    def perturb(self, image, label, sess, values,max_internal_queries, noise):
        """Perturb an image.

        Args:
            image: numpy array of size [1, 299, 299, 3], an original image
            label: numpy array of size [1], the label of the image (or target label)
            index: int, the index of the image
            sess: TensorFlow session

        Returns:
            adv_image: numpy array of size [1, 299, 299, 3], an adversarial image
            num_queries: int, the number of queries
            success: bool, True if attack is successful
        """
        # Set random seed by index for the reproducibility
        # np.random.seed(index)

        # Class variables
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.channels = image.shape[2]
        # print('========+> the label is', label)
        # Local variables
        adv_image = np.copy(image)
        num_queries = 0

        if values is not None:
            self.block_size = values['block_size']
            num_queries = values['num_queries']

        block_size = self.block_size

        upper_left = [0, 0]
        if self.no_hier:
          lower_right = [self.width,self.height]
          block_size = 1
        else:
          lower_right = [256, 256]

        # Split an image into a set of blocks
        blocks = self._split_block(upper_left, lower_right, block_size, self.num_channels)

        # Initialize a noise to -epsilon
        if noise is None:
            if not self.no_hier:
                noise = -self.epsilon*np.ones([1, 256, 256, self.channels], dtype=np.float32)
            else:
                noise = -self.epsilon*np.ones([1, self.width, self.height, self.channels], dtype=np.float32)


        # Construct a batch
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
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
                print("[STATS][L2] rate = {:.5g}, cost = {}, size = {}, loss = {:.5g}, time = {:.5g}".
                      format(i/num_batches, num_queries, block_size, loss, time_end-time_beg))
                sys.stdout.flush()
                time_beg = time.time()
                # Pick a mini-batch
                bstart = i*batch_size
                bend = min(bstart + batch_size, num_blocks)
                blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
                # Run local search algorithm on the mini-batch
                noise, queries, loss, success = self.local_search.perturb(
                    image, noise, label, sess, blocks_batch, num_queries)
                time_end = time.time()
                num_queries += queries
                internal_queries += queries
                # tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
                #     block_size, i, loss, num_queries))
                
                # setting values to transmit
                values = {'batch':i, 'block_size':block_size, 'curr_order':curr_order,
                          'num_queries': num_queries, 'loss':loss}

                # Generate an adversarial image
                # print('Going for the perturbation')
                adv_image = self._perturb_image(image, noise)
                # If query count exceeds the maximum queries, then return False
                if internal_queries > max_internal_queries:
                    return adv_image, num_queries, False, values, noise
                
                if num_queries > self.max_queries:
                    return adv_image, num_queries, True, values, noise
                # If attack succeeds, return True
                if success:
                    return adv_image, num_queries, True, values, noise

            # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
            if not self.no_hier and block_size >= 2:
                block_size //= 2
                blocks = self._split_block(upper_left, lower_right, block_size, self.num_channels)
                num_blocks = len(blocks)
                batch_size = self.batch_size if self.batch_size > 0 else num_blocks
                curr_order = np.random.permutation(num_blocks)
            # Otherwise, shuffle the order of the batch
            else:
                curr_order = np.random.permutation(num_blocks)
