# 700的数据

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

max_nepochs = 12 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
display = 10  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).
sample_path = './samples'
checkpoint_path = './checkpoints'
restore = ''   # Model snapshot to restore from

lambda_g = 0.1    # Weight of the classification loss
gamma_decay = 0.5 # Gumbel-softmax temperature anneal rate

train_data = {
    'batch_size': 64,
    #'seed': 123,
    'datasets': [
        {
            'files': './data/gender_data/all700.train.text',
            'vocab_file': './data/gender_data/vocab700',
            'data_name': ''
        },
        {
            'files': './data/gender_data/all700.train.label',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = './data/gender_data/all700.test.text'
test_data['datasets'][1]['files'] = './data/gender_data/all700.test.label'


ftrain_data = {
    'batch_size': 64,
    'datasets': [
        {
            'files': './data/gender_data/f700.train.text',
            'vocab_file': './data/gender_data/vocab700',
            'data_name': ''
        },
        {
            'files': './data/gender_data/f700.train.label',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'female_train'
}

ftest_data = copy.deepcopy(ftrain_data)
ftest_data['datasets'][0]['files'] = './data/gender_data/f700.test.text'
ftest_data['datasets'][1]['files'] = './data/gender_data/f700.test.label'



mtrain_data = {
    'batch_size': 64,
    'datasets': [
        {
            'files': './data/gender_data/m700.train.text',
            'vocab_file': './data/gender_data/vocab700',
            'data_name': ''
        },
        {
            'files': './data/gender_data/m700.train.label',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'male_train'
}

mtest_data = copy.deepcopy(mtrain_data)
mtest_data['datasets'][0]['files'] = './data/gender_data/m700.test.text'
mtest_data['datasets'][1]['files'] = './data/gender_data/m700.test.label'



model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 50,
        'max_decoding_length_infer': 50,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    
    'convnet':{
        # (1) Conv layers
        "num_conv_layers": 1,
        "filters": 128,
        "kernel_size": [3, 4, 5],
        "conv_activation": "relu",
        "conv_activation_kwargs": None,
        "other_conv_kwargs": None,
        # (2) Pooling layers
        "pooling": "MaxPooling1D",
        "pool_size": None,
        "pool_strides": 1,
        "other_pool_kwargs": None,
        # (3) Dense layers
        "num_dense_layers": 1,
        "dense_size": 128,
        "dense_activation": "identity",
        "dense_activation_kwargs": None,
        "final_dense_activation": None,
        "final_dense_activation_kwargs": None,
        "other_dense_kwargs": None,
        # (4) Dropout
        "dropout_conv": [1],
        "dropout_dense": [],
        "dropout_rate": 0.5,
        # (5) Others
        "name": "conv1d_network"
    },
    
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
