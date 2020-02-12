# import cv2
import heapq
import math
import numpy as np
import sys
import tensorflow as tf
import time
from PIL import Image
import pandas as pd

class Objfun(object):
    def __init__(self):
        self.nf = 0
        self.xs = []
        self.fs = []

    def __call__(self, x, y):
        self.nf += len(x)
        self.xs.append(x.copy())
        self.fs.append(y)

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
        
class LocalSearchHelper_2(object):
    """A helper for local search algorithm.
    Note that since heapq library only supports min heap, we flip the sign of loss function.
    """

    def __init__(self, model, args, **kwargs):
        """Initalize local search helper.

        Args:
          model: TensorFlow model
          loss_func: str, the type of loss function
          epsilon: float, the maximum perturbation of pixel value
        """
        # Hyperparameter setting
        self.epsilon = args.epsilon
        self.max_iters = args.max_iters
        self.targeted = args.targeted
        self.loss_func = args.loss_func
        self.max_queries = args.max_queries
        self.no_hier = args.no_hier
        # Network setting
        self.model = model

        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None])
        self.test_in = tf.placeholder(tf.float32, (None, args.dim_image, args.dim_image, args.num_channels), 'x')
        probs = model.predict(self.test_in)
        self.preds = tf.argmax(probs, axis=1)
        self.logits = tf.math.log(probs)

        batch_num = tf.range(0, limit=tf.shape(probs)[0])
        indices = tf.stack([batch_num, self.y_input], axis=1)
        # indices_sum = tf.stack([batch_num, self.y_input], axis=1)
        ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
        # ground_truth_probs_sum = tf.gather_nd(params=probs, indices=indices)
        top_2 = tf.nn.top_k(probs, k=2)
        max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
        max_indices = tf.stack([batch_num, max_indices], axis=1)
        max_probs = tf.gather_nd(params=probs, indices=max_indices)

        # self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)
        # self.real = tf.reduce_sum((self.tlab) * self.output, 1)
        # self.other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)
        # self.sum = tf.reduce_sum((1 - self.tlab) * self.output, 1)
        #
        # self.losses =
        if self.targeted:
            if self.loss_func == 'xent':
                self.losses = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
            elif self.loss_func == 'cw':
                # self.losses = tf.log(max_probs+1e-10) - tf.log(ground_truth_probs+1e-10)
                # self.losses = -tf.log(ground_truth_probs + 1e-10) + tf.log(tf.reduce_sum(probs, 1) - ground_truth_probs + 1e-10)
                # self.losses = -(ground_truth_probs + 1e-10) + (tf.reduce_sum(probs, 1) - ground_truth_probs + 1e-10)
                if self.no_hier:
                    self.losses = -(ground_truth_probs + 1e-10) + (max_probs + 1e-10)
                else:
                    # self.losses = -(ground_truth_probs + 1e-10) + (tf.reduce_sum(probs, 1) - ground_truth_probs + 1e-10)
                    self.losses = -tf.log(ground_truth_probs + 1e-10) + tf.log(tf.reduce_sum(probs, 1) - ground_truth_probs + 1e-10)
            else:
                tf.logging.info('Loss function must be xent or cw')
                sys.exit()
        else:
            if self.loss_func == 'xent':
                self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
            elif self.loss_func == 'cw':
                self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
            else:
                tf.logging.info('Loss function must be xent or cw')
                sys.exit()
 
    def _perturb_image(self, image, noise):
        """Given an image and a noise, generate a perturbed image.
        First, resize the noise with the size of the image.
        Then, add the resized noise to the image.

        Args:
          image: numpy array of size [1, 299, 299, 3], an original image
          noise: numpy array of size [1, 256, 256, 3], a noise

        Returns:
          adv_iamge: numpy array with size [1, 299, 299, 3], a perturbed image
        """
        # time_pre = time.time()
        # adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        resized_noise = tf.image.resize_images(np.array([noise[0,...]]), [self.width, self.height],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        # time_aft = time.time()
        # time_pret = time.time()
        resized_noise = resized_noise.eval()
        # print('MAX and min are', np.amax(resized_noise),np.amin(resized_noise), ' with eps ',
        #      ' equal to ', self.epsilon)
        # time_aftt = time.time()
        # print('--eval', time_aftt- time_pret)
        # resized_noise = np.resize([noise[0, ...]], [1, self.width, self.height, 3])
        adv_image = image + resized_noise#Image.open(noise[0, ...]).resize((self.width, self.height), Image.NEAREST)

        if self.width != 96:
          adv_image = np.clip(adv_image, -0.5, 0.5)
        else:
          adv_image = np.clip(adv_image, -1, 1)
        # print('Type adv', type(adv_image))
        # print('Type adv shape', adv_image.shape)
        return adv_image

    def _flip_noise(self, noise, block):
        """Flip the sign of perturbation on a block.
        Args:
          noise: numpy array of size [1, 256, 256, 3], a noise
          block: [upper_left, lower_right, channel], a block

        Returns:
          noise: numpy array of size [1, 256, 256, 3], an updated noise
        """
        noise_new = np.copy(noise)
        upper_left, lower_right, channel = block
        noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
        return noise_new

    def perturb(self, image, noise, label, sess, blocks, tot_num_queries):
        """Update a noise with local search algorithm.
        
        Args:
          image: numpy array of size [1, 299, 299, 3], an original image
          noise: numpy array of size [1, 256, 256, 3], a noise
          label: numpy array of size [1], the label of image (or target label)
          sess: TensorFlow session
          blocks: list, a set of blocks
    
        Returns: 
          noise: numpy array of size [1, 256, 256, 3], an updated noise
          num_queries: int, the number of queries
          curr_loss: float, the value of loss function
          success: bool, True if attack is successful
        """
        # Class variables
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.channels = image.shape[2]
        # times = np.zeros((10,))
        # Local variables
        priority_queue = []
        num_queries = 0

        summary = Objfun()
        # time_before_check = time.time()
        # Check if a block is in the working set or not
        A = np.zeros([len(blocks)], np.int32)
        for i, block in enumerate(blocks):
            upper_left, _, channel = block
            x = upper_left[0]
            y = upper_left[1]
            # If the sign of perturbation on the block is positive,
            # which means the block is in the working set, then set A to 1
            if noise[0, x, y, channel] > 0:
                A[i] = 1
        # time_after_check = time.time()
        # times[0] = time_before_check-time_after_check
        # Calculate the current loss
        # print('Image is ', image.shape, type(image))
        # print('The label is ', label, type(label), label.shape)
        # print(self.preds)
        # losses = sess.run([self.losses], feed_dict={self.test_in: np.array([image]), self.y_input: label})[0]
        # print(self.preds)
        # preds = sess.run([self.preds], feed_dict={self.test_in: np.array([image]), self.y_input: label})[0]
        # losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: np.array([image]),
        #                                                                self.y_input: label})
        # print(losses, preds)
        # time_inti_batch_beg = time.time()
        image_batch = self._perturb_image(image, noise)
        label_batch = np.array([float(np.copy(label))])
        # time_inti_batch_end = time.time()
        # times[1] = time_inti_batch_end - time_inti_batch_beg
        # print('Label Batch', image_batch, type(image_batch), image_batch.shape)
        # print('Label Batch', label_batch, type(label_batch), label_batch.shape)
        # time_single_eval_beg = time.time()
        losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: image_batch,
                                                                       self.y_input: label_batch})
        # time_single_eval_end = time.time()
        # times[2] = time_single_eval_end - time_single_eval_beg
        summary(A,losses)
        num_queries += 1
        curr_loss = losses[0]

        # Early stopping
        if self.targeted:
            if preds == label:
                return noise, num_queries, curr_loss, True
        else:
            if preds != label:
                return noise, num_queries, curr_loss, True
        # time_beg_main_loop = time.time()
        # Main loop
        for iteration in range(self.max_iters):
            # print("[STATS][L3] iter = {}, cost = {}, loss = {:.5g}, maxel = {}".
            #       format(iteration, num_queries,  curr_loss, preds[0]))
            # sys.stdout.flush()
            # Lazy greedy insert
            indices,  = np.where(A==0)

            batch_size = 100
            num_batches = int(math.ceil(len(indices)/batch_size))
            # time_beg_adding_nodes = time.time()
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))

                image_batch = np.zeros([bend-bstart, self.width, self.height, self.channels], np.float32)
                if self.no_hier:
                    noise_batch = np.zeros([bend-bstart, self.width, self.height, self.channels], np.float32)
                else:
                    noise_batch = np.zeros([bend-bstart, 256, 256, self.channels], np.float32)
                label_batch = np.tile(label, bend-bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

                losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: image_batch,
                                                                               self.y_input: label_batch})

                # Early stopping
                success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    print('Successfull at the lazy greedy insert')
                    return noise, num_queries, curr_loss, True
                num_queries += bend-bstart

                if tot_num_queries+num_queries>self.max_queries:
                    noise[0, ...] = noise_batch[0, ...]
                    curr_loss = losses[0]
                    return noise, num_queries, curr_loss, False

                # Push into the priority queue
                for i in range(bend-bstart):
                    idx = indices[bstart+i]
                    margin = losses[i]-curr_loss
                    heapq.heappush(priority_queue, (margin, idx))
            # time_end_adding_nodes = time.time()
            # times[3] = time_end_adding_nodes - time_beg_adding_nodes
            # Pick the best element and insert it into the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 1
            # time_beg_working_set = time.time()
            # Add elements into the working set
            while len(priority_queue) > 0:
                # Pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)

                # Re-evalulate the element
                image_batch = self._perturb_image(image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)

                losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: image_batch,
                                                                               self.y_input: label_batch})
                num_queries += 1
                margin = losses[0]-curr_loss

                # If the cardinality has not changed, add the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin > 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 1
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))
            # time_end_working_set = time.time()
            # times[4] = time_end_working_set - time_beg_working_set 
            priority_queue = []
            # Lazy greedy delete
            indices,  = np.where(A==1)

            batch_size = 100
            num_batches = int(math.ceil(len(indices)/batch_size))
            # time_beg_delete_set = time.time()
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))
                image_batch = np.zeros([bend-bstart, self.width, self.height, self.channels], np.float32)
                if self.no_hier:
                    noise_batch = np.zeros([bend-bstart, self.width, self.height, self.channels], np.float32)
                else:
                    noise_batch = np.zeros([bend-bstart, 256, 256, self.channels], np.float32)
                label_batch = np.tile(label, bend-bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i+1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i+1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])

                losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: image_batch,
                                                                               self.y_input: label_batch})

                # Early stopping
                success_indices,  = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    return noise, num_queries, curr_loss, True
                num_queries += bend-bstart

                if tot_num_queries+num_queries>self.max_queries:
                    noise[0, ...] = noise_batch[0, ...]
                    curr_loss = losses[0]
                    return noise, num_queries, curr_loss, False

                # Push into the priority queue
                for i in range(bend-bstart):
                    idx = indices[bstart+i]
                    margin = losses[i]-curr_loss
                    heapq.heappush(priority_queue, (margin, idx))
            # time_end_delete_set = time.time()
            # times[5] = time_end_delete_set - time_beg_delete_set 
            # Pick the best element and remove it from the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 0

            # Delete elements into the working set
            # time_beg_deleting_set = time.time()
            while len(priority_queue) > 0:
                # pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)

                # Re-evalulate the element
                image_batch = self._perturb_image(image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)

                losses, preds = sess.run((self.losses, self.preds), feed_dict={self.test_in: image_batch,
                                                                             self.y_input: label_batch})
                num_queries += 1
                margin = losses[0]-curr_loss

                # If the cardinality has not changed, remove the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin >= 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 0
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))
            priority_queue = []
        return noise, num_queries, curr_loss, False
