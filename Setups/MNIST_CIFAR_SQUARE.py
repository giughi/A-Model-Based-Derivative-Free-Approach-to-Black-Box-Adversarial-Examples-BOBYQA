"""
python Setups/Generation_Comparision_norm_robust.py --dataset=cifar

"""
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import random
import time

import pickle

from Setups.Data_and_Model.setup_cifar import CIFAR, CIFARModel
from Setups.Data_and_Model.setup_mnist import MNIST, MNISTModel
# from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel

# from Attack_Code.BOBYQA.BOBYQA_Attack_4_b import BlackBox_BOBYQA

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image
# python Setups/Generation_Comparision_norm_robust_SQUARE_Attack.py  --Adversary_trained=True --dataset=cifar10 --eps=0.1

# In[2]:
flags = tf.app.flags
flags.DEFINE_float('eps', 0.1, 'epsilon')
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', False, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Description for how to save the results')
flags.DEFINE_string('attack_type', '', 'attack used ')
FLAGS = flags.FLAGS

def generate_data(data, samples, targeted=True, start=0):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    for i in range(samples):
        if targeted:
            seq = range(data.test_labels.shape[1])
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start+i])
                true_ids.append(start+i)
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            labels.append(data.test_labels[start+i])
            true_ids.append(start+i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)

    return inputs, targets, labels, true_ids


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

def loss(y, logits, targeted=True):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        # print('========== y', y, len(y), len(logits))
        if targeted:
            targ_loss = logits[np.argmax(y)]
            # print('targ_loss',targ_loss)
            preds_correct_class = np.amax(logits)
            diff = preds_correct_class  - targ_loss  # difference between the correct class and all other classes
            return np.array([diff])
        else:
            preds_correct_class = np.sort(logits)
            diff = logits[y][0] - preds_correct_class[-2]  # difference between the correct class and all other classes
            return np.array(diff)


def square_attack_linf(sess, model, x, y, eps, n_iters, p_init, print_every=200):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = -0.5, 0.5 
    h, w, c = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    test_in = tf.placeholder(tf.float32, (1, h, w, c), 'x')
    test_pred = model.predict(test_in)
                    
        
    # Vertical stripes initialization
    x_best = np.clip(x + np.random.choice([-eps, eps], size=[x.shape[0], 1, w, c]), min_val, max_val)
    # print(x_best.shape)
    # print('y', y[0])
    logits = sess.run(test_pred, feed_dict={test_in: x_best})[0]
    margin_min = loss(y[0], logits)
    # print('logits', logits)
    print(margin_min)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query
    # print('n_queries', n_queries)
    time_start = time.time()
    # metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr = x, x_best
        y_curr, margin_min_curr = y, margin_min
        deltas = x_best_curr - x_curr

        p = 1/n_features#p_selection(p_init, i_iter, n_iters)
        
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            x_best_curr_window = x_best_curr[i_img, center_h:center_h+s, center_w:center_w+s, :]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, center_h:center_h+s, center_w:center_w+s, :], 
                                        min_val, max_val)
                                - x_best_curr_window) 
                         < 10**-7) == c*s*s:
                # the updates are the same across all elements in the square
                deltas[i_img, center_h:center_h+s, center_w:center_w+s, :] = np.random.choice([-eps, eps], size=[s, s, c])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = sess.run(test_pred, feed_dict={test_in: x_new})[0]
        margin = loss(y_curr[0], logits)
        # print('margin',margin, 'logits', logits)
        # print(margin)
        idx_improved = margin < margin_min_curr
        # print('====idx', margin_min[idx_to_fool], margin_min)
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = (idx_improved) * x_new + (~idx_improved) * x_best_curr
        n_queries[idx_to_fool] += 1

        # different metrics to keep track of
        acc = margin_min[0]
        time_total = time.time() - time_start
        if np.mod(i_iter, print_every)==0:
            print('[L1] {}: margin={}  (n_ex={}, eps={:.3f}, {:.2f}s)'.
                format(i_iter+1, acc, x.shape[0], eps, time_total))

        if acc<=0:
            break
    return n_queries, x_best


class args_CLASS(object): #= vars(parser.parse_args())
    dataset = 'cifar10'            # choices=["mnist", "cifar10", "imagenet"], default="mnist")
    save = './Results/'
    numimg = 0                  # type=int, default=0, help = "number of test images to attack")
    maxiter = 0                  # type=int, default=0, help = "set 0 to use default value")
    print_every = 100                # type=int, default=100, help = "print objs every PRINT_EVERY iterations")
    early_stop_iters = 100   # type=int, default=100, help = "print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    firstimg = 0                  # type=int, default=0)
    untargeted = True               # action='store_true')
    use_resize = False              # action='store_true', help = "resize image (only works on imagenet!)")
    seed = 1216               # type=int, default=1216)


if __name__ == '__main__':
    args = args_CLASS()
    args.dataset = FLAGS.dataset

    if args.dataset == 'mnist':
        args.save = './Results/MNIST'
    elif args.dataset == 'cifar10':
        args.save = './Results/CIFAR'
    if args.maxiter == 0:
        if args.dataset == "mnist":
            args.maxiter = 3000
        else:
            args.maxiter = 3000

    # setup random seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    args_pd = {}
    for elem in dir(args):
        if elem[:2] != '__':
            args_pd[elem] = eval('args.'+elem)

    args = args_pd

    args['use_resize'] = False
    args['attack'] = 'BOBYQA'

    args['print_every'] = FLAGS.print_every
    args['max_iter'] = FLAGS.max_evals
    args['untargeted'] = False
    

    BATCH = 50
    use_log = False
    vector_eval = []
    ORDERED_DOMAIN = True
    img_dist = False
    mix_dist = True
    MAX_EVAL = FLAGS.max_evals

    global_summary = []

    L_inf_var = FLAGS.eps
    list_dist = []
    list_gene = []
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4,
                                        inter_op_parallelism_threads=4)

    if FLAGS.dataset == 'mnist':
        saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
    else:
        saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description

    if FLAGS.Adversary_trained:
        case = '_distilled'
    else:
        case ='_normal'
    
    saving_name_dist = saving_dir + 'square_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals)+ case +'.txt'

    with tf.Session(config=session_conf) as sess:
    #     use_log = args['use_zvalue']
        # load network
        print('Loading model', args['dataset'])
        if args['dataset'] == "mnist":
            if FLAGS.Adversary_trained:
                data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist-distilled-100", sess, use_log)
            else:
                data, model = MNIST(), MNISTModel(main_dir + "/Models/mnist", sess, use_log)
        elif args['dataset'] == "cifar10":
            if FLAGS.Adversary_trained:
                data, model = CIFAR(), CIFARModel(main_dir + "/Models/cifar-distilled-100", sess, use_log)
            else:
                data, model = CIFAR(), CIFARModel(main_dir + "/Models/cifar", sess, use_log)


        print('Done...')
        args['numimg'] = FLAGS.test_size
        print('Using', args['numimg'], 'test images with ', L_inf_var,' energy')
        # load attack module
        
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        print('Generate data')
        all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, samples=args['numimg'], targeted=not args['untargeted'],
                                        start=args['firstimg'])
        print('Done...')
        # os.system("mkdir -p {}/{}".format(args['save'], args['dataset']))
        img_no = 0
        total_success = 0
        l2_total = 0.0        
        
        already_done_init = 0
        already_done = 0

        if os.path.exists(saving_name_dist):
            if os.path.getsize(saving_name_dist)>0:
                with open(saving_name_dist, "rb") as fp:
                    list_dist = pickle.load(fp)
                already_done_init = len(list_dist)

        
        for i in range(already_done_init,args['numimg']):
            inputs = all_inputs[i:i+1]
            targets = all_targets[i:i+1]
            labels = all_labels[i:i+1]
            print("true labels:", np.argmax(labels), labels)
            print("target:", np.argmax(targets), targets)
            # test if the image is correctly classified
            original_predict = model.model.predict(inputs)
            original_predict = np.squeeze(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("original probabilities:", original_prob[-1:-6:-1])
            print("original classification:", original_class[-1:-6:-1])
            print("original probabilities (most unlikely):", original_prob[:6])
            print("original classification (most unlikely):", original_class[:6])
            if original_class[-1] != np.argmax(labels):
                print("skip wrongly classified image no. {}, original class {}, classified as {}".format(i, np.argmax(labels), original_class[-1]))
                continue

            img_no += 1
            timestart = time.time()
            # adversarial generation
            # adv_dist, eval_costs_dist, summary = attack_dist.attack_batch(inputs, targets)
            
            result = square_attack_linf(sess, model, inputs, np.array(targets, dtype=int), L_inf_var, 3000, 0.1)
            
            end_time = time.time()

            query_count, adv_dist = result

            if len(adv_dist.shape) == 3:
                adv_dist = adv_dist.reshape((1,) + adv_dist.shape)

            timeend = time.time()
#             l2_distortion = np.sum((adv)**2)**.5
            adversarial_class = np.argmax(targets[0])
            
            adversarial_predict_dist = model.model.predict(inputs + adv_dist)
            adversarial_predict_dist = np.squeeze(adversarial_predict_dist)
            adversarial_prob_dist = np.sort(adversarial_predict_dist)
            adversarial_class_dist = np.argsort(adversarial_predict_dist)

            list_dist.append([query_count, original_prob[-1]-adversarial_prob_dist[-1],
                            np.argmax(labels), np.argmax(targets)])

            print("[STATS][L1] total = {}, seq = {}, batch = {}prev_class = {}, new_class = {}".format(img_no, i, BATCH, original_class[-1], adversarial_class))
            sys.stdout.flush()

            # here we save the results achieved

            with open(saving_name_dist, "wb") as fp:
                pickle.dump(list_dist, fp)

            print('saved')
