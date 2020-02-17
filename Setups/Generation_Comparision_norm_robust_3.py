"""
python Setups/Generation_Comparision_norm_robust_2.py --dataset=mnist --Adversary_trained=False

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
import inspect
import pickle

from Setups.Data_and_Model.setup_cifar import CIFAR, CIFARModel
from Setups.Data_and_Model.setup_mnist import MNIST, MNISTModel

from Attack_Code.Combinatorial import attacks
from Attack_Code.Combinatorial.tools.imagenet_labels import *
from Attack_Code.Combinatorial.tools.utils import *
from Setups.Data_and_Model.setup_inception_2 import ImageNet, InceptionModel
from Attack_Code.GenAttack import utils
# from Setups.Data_and_Model.setup_inception import ImageNet, InceptionModel
# from Attack_Code.BOBYQA.BOBYQA_Attack_4_b import BlackBox_BOBYQA

# uploading attacks
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)
  print('attack: ', attack)
  print('sys.modules', sys.modules[__name__])


dir_path = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.abspath(os.path.join(dir_path, os.pardir))



# In[2]:
flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', False, ' Use the adversarially trained nets')
flags.DEFINE_string('description', 'CONVALIDATION', 'Description for how to save the results')

# flags for combinatorial attack
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
# flags.DEFINE_string('save_dir', default=main_dir+'/Results/Imagenet', help='directory to save results')
# flags.DEFINE_string('save_img', default=main_dir+'/Results/Imagenet/Images', help='store_true')
# flags.DEFINE_integer('sample_size', 1, help='Number of test images.')
# flags.DEFINE_bool('verbose', True, help='Print logs.')
flags.DEFINE_integer('batch', 50, help='Dimension of the sampling domain.')
flags.DEFINE_integer('img_index_start', default=0, help='first image')

flags.DEFINE_float('epsilon', 0.3, help='maximum L_inf distance threshold') #######
flags.DEFINE_integer('max_steps', 1000, help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, help='Reduced dimension for dimensionality reduction') #######
# flags.DEFINE_string('model', default='inception', help='model name') ######
flags.DEFINE_string('loss_func', default='cw', help='The type of loss function') ######
flags.DEFINE_integer('target', 704, help='target class. if not provided will be random')
flags.DEFINE_string('attack', default='ParsimoniousAttack_2', help='The type of attack')
flags.DEFINE_integer('max_queries', default=3000, help='The query limit') ##########
flags.DEFINE_bool('targeted', default=True, help='bool on targeted')
flags.DEFINE_integer('max_iters', 1, help='maximum iterations') ##########
flags.DEFINE_integer('block_size', default=32, help='blck size') ##########
flags.DEFINE_integer('batch_size', default=64, help='batch size') ##########
flags.DEFINE_bool('no_hier', default=True, help='bool on hierarchical attack') #########
flags.DEFINE_string('input_dir', default='', help='Directory from whih to take the inputs') ########
flags.DEFINE_integer('dim_image', default=299, help='Dimension of the image that we feed as an input')
flags.DEFINE_integer('num_channels', default=3, help='Channels of the image that we feed as an input')
# flags.DEFINE_list('L_inf_var', default=[0.4], help='string witht he energy bounds for the attack')

FLAGS = flags.FLAGS

def generate_data(data, samples, targeted=True, start=0, inception=False):
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
            if inception:
                # for inception, randomly choose 10 target classes
                seq = np.random.choice(range(1,1001), 10)
                # seq = [580] # grand piano
            else:
                # for CIFAR and MNIST, generate all target classes
                seq = range(data.test_labels.shape[1])

            # print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
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
    attack = 'black'            # choices=["white", "black"], default="white")
    numimg = 0                  # type=int, default=0, help = "number of test images to attack")
    maxiter = 0                  # type=int, default=0, help = "set 0 to use default value")
    print_every = 10                # type=int, default=100, help = "print objs every PRINT_EVERY iterations")
    early_stop_iters = 100   # type=int, default=100, help = "print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    firstimg = 0                  # type=int, default=0)
    binary_steps = 0                 # type=int, default=0)
    init_const = 0.0                # type=float, default=0.0)
    use_zvalue = False              # action='store_true')
    untargeted = True               # action='store_true')
    reset_adam = False              # action='store_true', help = "reset adam after an initial solution is found")
    use_resize = False              # action='store_true', help = "resize image (only works on imagenet!)")
    adam_beta1 = 0.9                # type=float, default=0.9)
    adam_beta2 = 0.999              # type=float, default=0.999)
    seed = 1216               # type=int, default=1216)
    solver = 'adam'             # choices=["adam", "newton", "adam_newton", "fake_zero"], default="adam")
    save_ckpts = ""                 # default="", help = "path to save checkpoint file")
    load_ckpt  = ""                 # default="", help = "path to numpy checkpoint file")
    start_iter = 0                  # default=0, type=int, help = "iteration number for start, useful when loading a checkpoint")
    init_size  = 32                 # default=32, type=int, help = "starting with this size when --use_resize")
    uniform    = True               # action='store_true', help = "disable importance sampling")

    # add some additional parameters
    # learning rate
    lr = 1e-2
    inception = False
    use_tanh = False
#     args['use_resize'] = False


if __name__ == '__main__':
    args = args_CLASS()
    args.dataset = FLAGS.dataset

    if args.dataset == 'mnist':
        args.save = './Results/MNIST'
    elif args.dataset == 'cifar10':
        args.save = './Results/CIFAR'
    if args.maxiter == 0:
        if args.attack == "white":
            args.maxiter = 1000
        else:
            if args.dataset == "imagenet":
                if args.untargeted:
                    args.maxiter = 1500
                else:
                    args.maxiter = 50000
            elif args.dataset == "mnist":
                args.maxiter = 3000
            else:
                args.maxiter = 1000
    if args.init_const == 0.0:
        if args.binary_steps != 0:
            args.init_const = 0.01
        else:
            args.init_const = 0.5
    if args.binary_steps == 0:
        args.binary_steps = 1
    # set up some parameters based on datasets
    if args.dataset == "imagenet":
        args.inception = True
        args.lr = 2e-3
        # args['use_resize'] = True
        # args['save_ckpts'] = True
    # for mnist, using tanh causes gradient to vanish
    if args.dataset == "mnist":
        args.use_tanh = False
    # when init_const is not specified, use a reasonable default
    if args.init_const == 0.0:
        if args.binary_search:
            args.init_const = 0.01
        else:
            args.init_const = 0.5
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
    args['max_iter'] = FLAGS.max_iters
    args['untargeted'] = False
    if args['dataset'] == 'mnist':
        L_inf_var_list = [0.4]
    else:
        # if FLAGS.Adversary_trained:
        #     L_inf_var_list = [0.1, 0.05, 0.0275, 0.025, 0.0225, 0.02]
        # else:
        #     L_inf_var_list = [0.1, 0.05, 0.02, 0.01]
        L_inf_var_list = [0.05]
    
    BATCH = 50
    use_log = False
    vector_eval = []
    ORDERED_DOMAIN = True
    img_dist = False
    mix_dist = True
    MAX_EVAL = FLAGS.max_evals


    for L_inf_var in L_inf_var_list:
        list_combi = []
        already_done = 0

        # Defining the name of the saving file
        if FLAGS.dataset == 'mnist':
            saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
        else:
            saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description

        if FLAGS.Adversary_trained:
            case = '_distilled_'
        else:
            case ='_normal'
        saving_name = saving_dir+'combi_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals)+ case +'.txt'
        
        # Uploading the previous results if there are
        already_done = 0
        if os.path.exists(saving_name):
            if os.path.getsize(saving_name)>0:
                with open(saving_name, "rb") as fp:
                    list_combi = pickle.load(fp)
                    already_done = len(list_combi)
                
        if args['numimg'] == 0:
            args['numimg'] = 1300
        print('Using', args['numimg'], 'test images with ', L_inf_var,' energy')
        print('==============> Already done', already_done)
        for i in range(already_done,args['numimg']):


            tf.reset_default_graph()
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=4,
                                          inter_op_parallelism_threads=4)

            with tf.Session(config=session_conf) as sess:
                is_inception = args['dataset'] == "imagenet"
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

                elif args['dataset'] == "imagenet":
                    data, model = ImageNet(), InceptionModel(sess, use_log)
                print('Done...')
                
                # load attack module

                FLAGS.epsilon= L_inf_var
                FLAGS.max_iters = args['max_iter']
                attack_class = getattr(sys.modules[__name__], FLAGS.attack)
                print(model)
                attack_combi = attack_class(model, FLAGS)

                random.seed(args['seed'])
                np.random.seed(args['seed'])
                print('Generate data')
                all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, samples=args['numimg'], targeted=not args['untargeted'],
                                                start=args['firstimg'], inception=is_inception)
                print('Done...')
                # os.system("mkdir -p {}/{}".format(args['save'], args['dataset']))
                img_no = 0
                total_success = 0
                l2_total = 0.0

                

                                    
                inputs = all_inputs[i:i+1]
                targets = all_targets[i:i+1]
                labels = all_labels[i:i+1]
                original_predict = model.model.predict(inputs)
                original_predict = np.squeeze(original_predict)
                original_prob = np.sort(original_predict)
                original_class = np.argsort(original_predict)
                print('classifications', original_class[-1] ,np.argmax(labels))
                if original_class[-1] != np.argmax(labels):
                    print("skip wrongly classified image no. {}, original class {}, classified as {}".format(i, np.argmax(labels), original_class[-1]))
                    continue

                img_no += 1
                timestart = time.time()
                # adversarial generation
                time_beg = time.time()
                adv_combi, eval_costs_combi, succ_combi = attack_combi.perturb(inputs[0], np.array([np.argmax(targets)]), 0, sess)
                time_end = time.time()
                if succ_combi:
                    print('[STATS][L0]Succesful attack with ', eval_costs_combi, 'queries')

                if len(adv_combi.shape) == 3:
                    adv_combi = adv_combi.reshape((1,) + adv_combi.shape)

                timeend = time.time()
                adversarial_class = np.argmax(targets[0])

                adversarial_predict_combi = model.model.predict(inputs + adv_combi)
                adversarial_predict_combi = np.squeeze(adversarial_predict_combi)
                adversarial_prob_combi = np.sort(adversarial_predict_combi)
                adversarial_class_combi = np.argsort(adversarial_predict_combi)

                list_combi.append([eval_costs_combi, original_prob[-1]-adversarial_prob_combi[-1],
                                np.argmax(labels), np.argmax(targets)])

                print("[STATS][L1] total = {}, seq = {}, batch = {}, prev_class = {}, new_class = {}, attack_time = {:.5g}".format(
                    img_no, i, BATCH, original_class[-1], adversarial_class, time_end-time_beg))
                sys.stdout.flush()

                # here we save the results achieved
                if FLAGS.dataset == 'mnist':
                    saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
                else:
                    saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description

                if FLAGS.Adversary_trained:
                    case = '_distilled_'
                else:
                    case ='_normal'
                with open(saving_dir+'combi_L_inf_'+str(L_inf_var)+'_max_eval_'+ str(FLAGS.max_evals)+ case +'.txt', "wb") as fp:
                    pickle.dump(list_combi, fp)

                print('saved')
