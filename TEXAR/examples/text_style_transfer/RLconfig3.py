"""Config
"""
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
            'files': './data/gender_data/all_train.text',
            'vocab_file': './data/gender_data/vocab',
            'data_name': ''
        },
        {
            'files': './data/gender_data/all_train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}
val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = './data/gender_data/all_dev.text'
val_data['datasets'][1]['files'] = './data/gender_data/all_dev.labels'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = './data/gender_data/all_test.text'
test_data['datasets'][1]['files'] = './data/gender_data/all_test.labels'


ftrain_data = {
    'batch_size': 64,
    'datasets': [
        {
            'files': './data/gender_data/female_train.text',
            'vocab_file': './data/gender_data/vocab',
            'data_name': ''
        },
        {
            'files': './data/gender_data/female_train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'female_train'
}


fval_data = copy.deepcopy(ftrain_data)
fval_data['datasets'][0]['files'] = './data/gender_data/female_dev.text'
fval_data['datasets'][1]['files'] = './data/gender_data/female_dev.labels'

ftest_data = copy.deepcopy(ftrain_data)
ftest_data['datasets'][0]['files'] = './data/gender_data/female_test.text'
ftest_data['datasets'][1]['files'] = './data/gender_data/female_test.labels'



mtrain_data = {
    'batch_size': 64,
    'datasets': [
        {
            'files': './data/gender_data/male_train.text',
            'vocab_file': './data/gender_data/vocab',
            'data_name': ''
        },
        {
            'files': './data/gender_data/male_train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'male_train'
}


mval_data = copy.deepcopy(ftrain_data)
mval_data['datasets'][0]['files'] = './data/gender_data/male_dev.text'
mval_data['datasets'][1]['files'] = './data/gender_data/male_dev.labels'


mtest_data = copy.deepcopy(mtrain_data)
mtest_data['datasets'][0]['files'] = './data/gender_data/male_test.text'
mtest_data['datasets'][1]['files'] = './data/gender_data/male_test.labels'



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
        'max_decoding_length_train': 30,
        'max_decoding_length_infer': 30,
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
