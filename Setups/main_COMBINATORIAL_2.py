"""
python Setups/main_COMBINATORIAL_2.py --input_dir=Data/ImageNet/images --test_size=1 \
    --eps=0.05 \
    --max_steps=500000 --output_dir=attack_outputs  \
    --target=704

"""

import argparse
import inspect
import json
import numpy as np
import os
from PIL import Image
import sys
sys.path.append("./")
import random
import tensorflow as tf

from Attack_Code.Combinatorial import attacks
from Attack_Code.Combinatorial.tools.imagenet_labels import *
from Attack_Code.Combinatorial.tools.utils import *
from Setups.Data_and_Model.setup_inception_2 import ImageNet, InceptionModel
from Attack_Code.GenAttack import utils

""" The collection of all attack classes """
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]

print('Dict',[x for x in attacks.__dict__])
print('Initial contained', [x for x in attacks.__dict__.values()])
print('ATTACK CLASES ', ATTACK_CLASSES)
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)
  print('attack: ', attack)
  print('sys.modules', sys.modules[__name__])

""" Arguments """
dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

IMAGENET_PATH =  main_dir + '/Data/ImageNet/images'
NUM_LABELS = 1000

parser = argparse.ArgumentParser()

# Directory

flags = tf.app.flags
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_string('save_dir', default=main_dir+'/Results/Imagenet', help='directory to save results')
flags.DEFINE_string('save_img', default=main_dir+'/Results/Imagenet/Images', help='store_true')
flags.DEFINE_integer('sample_size', 1, help='Number of test images.')
flags.DEFINE_bool('verbose', True, help='Print logs.')
flags.DEFINE_integer('batch', 50, help='Dimension of the sampling domain.')
flags.DEFINE_integer('img_index_start', default=0, help='first image')

flags.DEFINE_float('epsilon', 0.10, help='maximum L_inf distance threshold')
flags.DEFINE_integer('max_steps', 1000, help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, help='Reduced dimension for dimensionality reduction')
flags.DEFINE_string('model', default='inception', help='model name')
flags.DEFINE_string('loss_func', default='xent', help='The type of loss function')
flags.DEFINE_integer('seed', 1216, help='random seed')
flags.DEFINE_integer('target', 704, help='target class. if not provided will be random')
flags.DEFINE_string('attack', default='ParsimoniousAttack_2', help='The type of attack')
flags.DEFINE_integer('max_queries', default=100000, help='The query limit')
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations')
flags.DEFINE_integer('block_size', default=32, help='blck size')
flags.DEFINE_integer('batch_size', default=64, help='batch size')
flags.DEFINE_bool('no_hier', default=False, help='bool on hierarchical attack')
flags.DEFINE_string('input_dir', default='', help='Directory from whih to take the inputs')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')
flags.DEFINE_integer('dim_image', default=299, help='Dimension of the image that we feed as an input')
# flags.DEFINE_integer('target', default=704, help='Target that we want to attack')
FLAGS = flags.FLAGS


if __name__ == '__main__':
    # Set verbosity
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = ImageNet(FLAGS.input_dir)
    inputs, targets, reals, paths = utils.generate_data(dataset, FLAGS.sample_size)
    # print(inputs[0])
    #  Create a session
    with tf.Session() as sess:
        model = InceptionModel(sess, use_log=True)
        test_in = tf.placeholder(tf.float32, (None, 299, 299, 3), 'x')
        test_pred = tf.argmax(model.predict(test_in), axis=1)
        print('Module updated')
        # Create attack
        attack_class = getattr(sys.modules[__name__], FLAGS.attack)
        attack = attack_class(model, FLAGS)
        print('attack models updated')
        indices = np.arange(0, 2000)
        # Main loop
        count = 0
        index = FLAGS.img_index_start
        total_num_corrects = 0
        total_queries = []
        index_to_query = {}
        print('Beginning of the while loop of length', FLAGS.sample_size)
        while count < FLAGS.sample_size:
            print('Count', count)
            input_img = inputs[index]
            input_img_path = paths[index]
            print('Imgaes of type', type(input_img[0, 0, 0]))
            print('Images uploaded with shape', input_img.shape)
            print('Model pred', model.model.predict(np.array([input_img])))
            # if FLAGS.targeted:
            #     target_class = np.eye(len(targets[index]))[FLAGS.target + 1]
            # else:
            #     target_class = np.argmax(targets[index])
            target_class = np.array([FLAGS.target + 1])
            print('Target chosen')
            real_label = reals[index]
            orig_pred = sess.run([test_pred], feed_dict={test_in: np.array([input_img])})[0]
            print('Real = {}, Predicted = {}, Target = {}'.format(
                    real_label, orig_pred, target_class))
            if orig_pred != real_label:
                print('\t Skipping incorrectly classified image.')
                index += 1
                continue

            # orig_class = np.expand_dims(orig_pred, axis=0)
            orig_class = orig_pred

            count += 1

            # Run attack
            if FLAGS.targeted:
                adv_img, num_queries, success = attack.perturb(input_img, target_class, indices[index], sess)
            else:
                adv_img, num_queries, success = attack.perturb(input_img, orig_class, indices[index], sess)

            # Check if the adversarial image satisfies the constraint
            assert np.amax(np.abs(adv_img-input_img)) <= FLAGS.epsilon+1e-3
            assert np.amax(adv_img) <= +0.5+1e-3
            assert np.amin(adv_img) >= -0.5-1e-3
            p = sess.run(test_pred, feed_dict={test_in: [input_img]})[0]
            # Save the adversarial image
            if FLAGS.save_img:
                adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :]*255, np.uint8))
                adv_image.save(os.path.join(FLAGS.save_dir, '{}_adv.jpg'.format(indices[index])))

            # Logging
            if success:
                total_num_corrects += 1
                total_queries.append(num_queries)
                index_to_query[indices[index]] = num_queries
                average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
                median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
                success_rate = total_num_corrects/count
                # tf.logging.info('Attack succeeds, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: '
                #                 '{:.4f}'.format(label_to_name(p[0]), average_queries, median_queries, success_rate))
            else:
                index_to_query[indices[index]] = -1
                average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
                median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
                success_rate = total_num_corrects/count
                # tf.logging.info('Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: '
                #                 '{:.4f}'.format(label_to_name(p[0]), average_queries, median_queries, success_rate))

            index += 1
