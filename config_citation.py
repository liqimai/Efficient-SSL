import time
from os import cpu_count
from copy import deepcopy
import tensorflow as tf

default = {
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
        'alpha'             : 10,


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
                                    'alpha' : 10,
                                } for _ in range(2)],
        'optimizer'         : tf.train.AdamOptimizer,
        'learning_rate'     : 0.01,         # 'Initial learning rate.'
        'epochs'            : 200,          # 'Number of epochs to train.'
        'dropout'           : 0.5,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'


        # Filter as pre-processing
        'smooth_config'     :{
            'type'   : None,         # 'taubin' | None | 'ap_appro'
            'alpha'  : 10,          # alpha for AR filter
            'k'      : 2,            # k for RNM and RW filter
        },


        'logging'           : False,        # 'Weather or not to record log'
        'logdir'            : '',           # 'Log directory.''
        'name'              : None,         # 'name of the model. Serve as an ID of model.'

        'random_seed'       : int(time.time()),  # 'Random seed.'
        'threads'           : cpu_count(),  #'Number of threads'
        'train'             : True,
    }

repeat = 1


small_label_set ={
    # repeating times
    'repeating': repeat,

    # The default model configuration
    'default': deepcopy(default),

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':    [
        # MLP
        {
            'Model': 'MLP',
            'name' : 'MLP',
            'connection': 'ff',
        },
        # LP
        {
            'Model': 'LP',
            'name' : 'LP',
            'alpha': 100,
        },
        # GCN
        {
            'Model': 'IGCN', # GCN is a special case of IGCN
            'name' : 'GCN'
        },
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
small_label_set['default'].update({'train_size': 4})

large_label_set ={
    # repeating times
    'repeating': repeat,

    # The default model configuration
    'default': deepcopy(default),

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':    [
        # MLP
        {
            'Model': 'MLP',
            'name' : 'MLP',
            'connection': 'ff',
        },
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
    ]

    # IGCN
    + [
        # IGCN(RNM)
        {
            'Model': 'IGCN',
            'name' : 'IGCN_RNM',
            'connection': 'cc',
            'conv_config': [
                {
                    'conv': 'rnm',
                    'k': 5 // 2,
                },
                {
                    'conv': 'rnm',
                    'k': 5 // 2,
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
                    'alpha': 10//2,
                },
                {
                    'conv': 'ap',
                    'alpha': 10//2,
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
                'k': 5,
            }
        },
        # GLP(AR)
        {
            'Model': 'GLP',
            'name' : 'GLP_AR',
            'connection': 'ff',
            'smooth_config': {
                'type': 'ap_appro',
                'alpha': 10,
            }
        }
    ]
}
large_label_set['default'].update({'train_size': 20})
