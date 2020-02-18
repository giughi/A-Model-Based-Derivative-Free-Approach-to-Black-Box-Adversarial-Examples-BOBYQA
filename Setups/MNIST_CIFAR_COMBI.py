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


flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'model name')
flags.DEFINE_integer('test_size', 1, 'Number of test images.')
flags.DEFINE_integer('max_evals', 3000, 'Maximum number of function evaluations.')
flags.DEFINE_integer('print_every', 10, 'Every iterations the attack function has to print out.')
flags.DEFINE_integer('seed', 1216, 'random seed')
flags.DEFINE_bool('Adversary_trained', False, ' Use the adversarially trained nets')
flags.DEFINE_string('description', '', 'Description for how to save the results')

# flags for combinatorial attack
flags.DEFINE_string('asset_dir', default=main_dir+'/Attack_Code/Combinatorial/assets', help='directory assets')
flags.DEFINE_integer('batch', 50, help='Dimension of the sampling domain.')
flags.DEFINE_integer('img_index_start', default=0, help='first image')
flags.DEFINE_float('epsilon', 0.3, help='maximum L_inf distance threshold') #######
flags.DEFINE_integer('max_steps', 1000, help='Maximum number of iterations')
flags.DEFINE_integer('resize_dim', 96, help='Reduced dimension for dimensionality reduction') #######
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
    
    BATCH = 50
    use_log = False
    vector_eval = []
    ORDERED_DOMAIN = True
    img_dist = False
    mix_dist = True
    MAX_EVAL = FLAGS.max_evals

    L_inf_var = FLAGS.epsilon

    
    list_combi = []
    already_done = 0

    # Defining the name of the saving file
    if FLAGS.dataset == 'mnist':
        saving_dir = main_dir + '/Results/MNIST/' + FLAGS.description
    else:
        saving_dir = main_dir + '/Results/CIFAR/' + FLAGS.description

    if FLAGS.Adversary_trained:
        case = '_distilled'
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
            

    args['numimg'] = FLAGS.test_size
    print('Using', args['numimg'], 'test images with ', L_inf_var,' energy')
    print('==============> Already done', already_done)
    for i in range(already_done,args['numimg']):


        tf.reset_default_graph()
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
                                            start=args['firstimg'])
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
            with open(saving_name, "wb") as fp:
                pickle.dump(list_combi, fp)

            print('saved')

