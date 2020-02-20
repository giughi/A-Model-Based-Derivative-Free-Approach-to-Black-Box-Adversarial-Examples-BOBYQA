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

from Attack_Code.MNIST_CIFAR_boby_gen import BlackBox

dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))

from PIL import Image

# In[2]:
flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', False, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Description for how to save the results')
flags.DEFINE_string('attack_type', '', 'attack used ')
flags.DEFINE_float('eps', 0.1, 'perturbation energy')
flags.DEFINE_bool('internal_summaries', False, 'perturbation energy')
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
    q = BATCH

    global_summary = []

    L_inf_var = FLAGS.eps

    list_boby = []
    list_gene = []
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4,
                                        inter_op_parallelism_threads=4)

    with tf.Session(config=session_conf) as sess:
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
                
        attack_dist = BlackBox(sess, model, batch_size=BATCH, max_iterations=args['maxiter'], print_every=args['print_every'],
                        early_stop_iters=args['early_stop_iters'], confidence=0, targeted=not args['untargeted'],
                        use_resize=args['use_resize'],L_inf=L_inf_var, BOBYQA=True,
                        max_eval = MAX_EVAL, q = q)

        # attack_gene = BlackBox_BOBYQA(sess, model, batch_size=6, max_iterations=args['maxiter'], print_every=args['print_every'],
        #               early_stop_iters=args['early_stop_iters'], confidence=0, targeted=not args['untargeted'], use_log=use_log,
        #               use_tanh=args['use_tanh'], use_resize=args['use_resize'],L_inf=L_inf_var, rank = 3, ordered_domain=True,
        #               image_distribution = False, mixed_distributions=True,GenAttack=True,max_eval = MAX_EVAL)

        print('Generate data')
        all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, samples=args['numimg'], targeted=not args['untargeted'],
                                        start=args['firstimg'])
        print('Done...')
        
        img_no = 0
        total_success = 0
        
        if FLAGS.dataset == 'mnist':
            saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
        else:
            saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description
        

        if FLAGS.Adversary_trained:
            case = '_distilled'
        else:
            case ='_normal'

        saving_name_boby = saving_dir+'boby_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) + case + '.txt'
        # saving_name_gene = saving_dir+'gene_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) +'.txt'
        
        already_done_init = 0
        already_done = 0


        if os.path.exists(saving_name_boby):
            if os.path.getsize(saving_name_boby)>0:
                with open(saving_name_boby, "rb") as fp:
                    list_boby = pickle.load(fp)
                already_done_init = len(list_boby)

        already_done = already_done_init

        if already_done_init>0:
            found = False
            while not found:
                lab = list_boby[-1][2]
                tar = list_boby[-1][3] 
                # if FLAGS.attack_type == 'gene':    
                #     lab = list_gene[-1][2]
                #     tar = list_gene[-1][3] 
                
                lab_ = np.argmax(all_labels[already_done_init-1:already_done_init])
                tar_ = np.argmax(all_targets[already_done_init-1:already_done_init])
                print(already_done, lab, tar, lab_, tar_)
                already_done_init +=1
                if lab == lab_ and tar==tar_:
                    found = True
                    already_done_init -= 1

        print('We have done ', already_done, ' attacks but are at iteration ', already_done_init)
        
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
            adv_dist, eval_costs_dist, summary = attack_dist.attack_batch(inputs, targets)
            # elif FLAGS.attack_type == 'gene':
            #     adv_gene, eval_costs_gene, summary = attack_gene.attack_batch(inputs, targets)

            if len(adv_dist.shape) == 3:
                adv_dist = adv_dist.reshape((1,) + adv_dist.shape)
            # if FLAGS.attack_type == 'gene':
            #     if len(adv_gene.shape) == 3:
            #         adv_gene = adv_gene.reshape((1,) + adv_gene.shape)

            timeend = time.time()
            adversarial_class = np.argmax(targets[0])

            adversarial_predict_dist = model.model.predict(inputs + adv_dist)
            adversarial_predict_dist = np.squeeze(adversarial_predict_dist)
            adversarial_prob_dist = np.sort(adversarial_predict_dist)
            adversarial_class_dist = np.argsort(adversarial_predict_dist)
            # if FLAGS.attack_type == 'gene':
            #     adversaria# python Setups/Generation_Comparision_norm_robust.py --max_evals=1400 --attack_type=boby --dataset=mnistl_predict_gene = model.model.predict(inputs + adv_gene)
            #     adversarial_predict_gene = np.squeeze(adversarial_predict_gene)
            #     adversarial_prob_gene = np.sort(adversarial_predict_gene)
            #     adversarial_class_gene = np.argsort(adversarial_predict_gene)

            list_boby.append([eval_costs_dist, original_prob[-1]-adversarial_prob_dist[-1],
                            np.argmax(labels), np.argmax(targets)])
            # if FLAGS.attack_type == 'gene':
            #     list_gene.append([eval_costs_gene, original_prob[-1]-adversarial_prob_gene[-1],
            #                     np.argmax(labels), np.argmax(targets)])

            print("[STATS][L1] total = {}, seq = {}, batch = {}prev_class = {}, new_class = {}".format(img_no, i, BATCH, original_class[-1], adversarial_class))
            sys.stdout.flush()

            if FLAGS.internal_summaries:
                global_summary.append(summary)
                with open(saving_dir+'summary_dist_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) +'_batch_'+str(BATCH)+'.txt', "wb") as fp:
                    pickle.dump(global_summary, fp)

            with open(saving_name_boby , "wb") as fp:
                    pickle.dump(list_boby, fp)


            # if FLAGS.attack_type == 'gene':
            #     with open(saving_dir+'summary_'+str(args['use_resize'])+'gene_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals) +'.txt', "wb") as fp:
            #         pickle.dump(summary, fp)

            print('saved')


# python Setups/Generation_Comparision_norm_robust.py --max_evals=1400 --attack_type=boby --dataset=mnist --test_size=20
