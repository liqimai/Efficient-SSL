from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config
import argparse
import pprint
import tensorflow as tf

configuration = {
    # repeat times
    'repeating'             : 1,

    # The default model configuration
    'default':{
        # Dataset and split
        'dataset'           : 'cora',       # 'Dataset (cora | citeseer | pubmed | large_cora | nell.0.1 | nell.0.01 | nell.0.001)
        'shuffle'           : True,         # random split or not
        'train_size'        : 4,            # if train_size is a number, then use TRAIN_SIZE labels per class.
        'validation_size'   : 500,          # 'Use VALIDATION_SIZE data to train model'
        'validate'          : False,        # Whether use validation set
        'test_size'         : None,         # If None, all rest are test set
        'feature'           : 'bow',        # 'bow' | 'tfidf' | 'none'.

        'Model'             : 'IGCN',       # 'LP', 'IGCN', 'GLP'， 'MLP'

        # 'alpha' in LP
        'alpha'             : 0.1,


        # Neural Network Setting
        'connection'        : 'cc',
        # A string contains only char "c" or "f"
        # "c" stands for convolution.
        # "f" stands for fully connected.

        'layer_size'        : [16],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        # graph conv in layers
        'conv_config'       :   [{
                                    'conv'  : 'rnm', # rnm, rw, ap
                                    'k'     : 1,
                                    'alpha' : 1/10,
                                } for _ in range(2)],
        'optimizer'         : tf.train.AdamOptimizer,
        'learning_rate'     : 0.01,         # 'Initial learning rate.'
        'epochs'            : 300,          # 'Number of epochs to train.'
        'dropout'           : 0.5,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'


        # Filter as pre-processing
        'smooth_config'     :{
            'type'   : None,         # 'taubin' | None | 'ap_appro'
            'alpha'  : 0.1,          # alpha for AR filter
            'k'      : 2,            # k for RNM and RW filter
        },


        'logging'           : False,        # 'Weather or not to record log'
        'logdir'            : '',           # 'Log directory.''
        'name'              : None,         # 'name of the model. Serve as an ID of model.'

        'random_seed'       : int(time.time()),  # 'Random seed.'
        'threads'           : cpu_count(),  #'Number of threads'
        'train'             : True,
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':
    [
        # LP
        {
            'Model': 'LP',
            'name' : 'LP',
            'alpha': 100,
        },
        # GCN
        {
            'Model': 'IGCN',
            'name' : 'GCN'
        },
        # MLP
        {
            'Model': 'MLP',
            'name' : 'MLP',
            'connection': 'ff',
        }
    ]

    + [
        # IGCN(RNM)
        {
            'Model': 'IGCN',
            'name' : 'IGCN_RNM',
            'connection': 'cc',
            'conv_config': [
                {
                    'conv': 'rnm',
                    'k': 10 // 2,
                },
                {
                    'conv': 'rnm',
                    'k': 10 // 2,
                }],
        },

        # IGCN(RW)
        {
            'Model': 'IGCN',
            'name' : 'IGCN_RW',
            'connection': 'cc',
            'conv_config': [
                {
                    'conv': 'rw',
                    'k': 20 // 2,
                },
                {
                    'conv': 'rw',
                    'k': 20 // 2,
                }],
        },
        # IGCN(AR)
        {
            'Model': 'IGCN',
            'name' : 'IGCN_AR',
            'connection': 'cc',
            'conv_config': [
                {
                    'conv': 'ap',
                    'alpha': 20//2,
                },
                {
                    'conv': 'ap',
                    'alpha': 20//2,
                }]
        }
    ]

    # GLP
    + [
        # GLP(RNM)
        {
            'Model': 'GLP',
            'name' : 'GLP_RNM',
            'connection': 'ff',
            'smooth_config': {
                'type': 'rnm',
                'k': 10,
            }
        },
        # GLP(RW)
        {
            'Model': 'GLP',
            'name' : 'GLP_RW',
            'connection': 'ff',
            'smooth_config': {
                'type': 'rw',
                'k': 20,
            }
        },
        # GLP(AR)
        {
            'Model': 'GLP',
            'name' : 'GLP_AR',
            'connection': 'ff',
            'smooth_config': {
                'type': 'ap_appro',
                'alpha': 20,
            }
        }
    ]
}


# Parse args
parser = argparse.ArgumentParser(description=(
    "Implementation for IGCN and GLP model in our paper\n"
    """"Label Efficient Semi-Supervised Learning via Graph Filtering. (CVPR-19)"\n"""
    "Most configuration are specified in config.py, please read it and modify it as you want."))
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--dataset", type=str, help='dataset name')
parser.add_argument("--train_size", type=str, help='labels per class')
parser.add_argument("--repeat", type=int, help='repeat times')
parser.add_argument("--validate", type=bool, default=None, help='True or False, if use validation set or not')
parser.add_argument("--pset", type=str, help='Parameter Set, e.g. config_citation.large')
parser.add_argument("--learning_rate", type=float, help='learning rate')
parser.add_argument("--seed", type=int, help='Random seed.')
parser.add_argument("--layer-size", type=eval, help='a python list of hidden layer widths')
args = parser.parse_args()
print(args)

if args.pset is not None:
    configuration = eval(args.pset)
if args.dataset is not None:
    configuration['default']['dataset'] = args.dataset
if args.train_size is not None:
    configuration['default']['train_size'] = eval(args.train_size)
if args.repeat is not None:
    configuration['repeating']=args.repeat
if args.learning_rate is not None:
    configuration['default']['learning_rate']=args.learning_rate
if args.seed is not None:
    configuration['default']['random_seed']=args.seed
if args.validate is not None:
    configuration['default']['validate'] = args.validate
if args.verbose is not None:
    configuration['default']['verbose'] = args.verbose
if args.layer_size is not None:
    configuration['default']['layer_size'] = args.layer_size
pprint.PrettyPrinter(indent=4).pprint(configuration)


def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config


configuration['model_list'] = list(map(set_default_attr, configuration['model_list']))
for model_config in configuration['model_list']:
    preprocess_model_config(model_config)

