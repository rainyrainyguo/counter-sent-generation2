import math
import os
import torch
    
    
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import SST_Dataset
from ctextgen.model import RNN_VAE

import argparse
import random
import time


parser = argparse.ArgumentParser(
    description='Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--model', default='yelpgen', metavar='',
                    help='choose the model: {`vae`, `ctextgen`,`yelpgen`}, (default: `ctextgen`)')

args = parser.parse_args()


mb_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2

#dataset = SST_Dataset()
'''
torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)
'''

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
from torchtext import data

#TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy',fix_length=30)
TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
LABEL = data.Field(sequential=False, unk_token=None)

print("loading dataset female_sent_obftrain_less100.tsv...")
train  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='female_sent_obftrain_less100.tsv',
        format='tsv',
        fields=[('Text', TEXT),('Label', LABEL)])[0]

TEXT.build_vocab(train, max_size=10000, vectors="fasttext.en.300d")
#TEXT.build_vocab(train, max_size=10000)
LABEL.build_vocab(train)

LABEL.vocab.stoi['1']=0
LABEL.vocab.stoi['2']=0
LABEL.vocab.stoi['3']=0
LABEL.vocab.stoi['4']=1
LABEL.vocab.stoi['5']=1




model = RNN_VAE(
    len(TEXT.vocab), h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=TEXT.vocab.vectors, freeze_embeddings=False,
    gpu=args.gpu
)



if args.gpu:
    model.load_state_dict(torch.load('models/{}.bin'.format(args.model)))
else:
    model.load_state_dict(torch.load('models/{}.bin'.format(args.model), map_location=lambda storage, loc: storage))

    
  
    
    
for i in range(3):
    print("---------------example-------------   ", i)
    # Samples latent and conditional codes randomly from prior
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)

    # Generate negative sample given z
    c[0, 0], c[0, 1] = 1, 0

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=0.5)
    sample_sent = ' '.join([TEXT.vocab.itos[i] for i in sample_idxs])

    #print('\nSentiment negative: {}'.format(dataset.idx2label(int(c_idx))))
    #print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))
    print('\nSentiment negative: {}'.format(int(c_idx)))
    print('Generated: {}'.format(sample_sent))
    
    
    # Generate positive sample from the same z
    c[0, 0], c[0, 1] = 0, 1

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=0.8)
    sample_sent = ' '.join([TEXT.vocab.itos[i] for i in sample_idxs])

    #print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
    print('\nSentiment positive: {}'.format(int(c_idx)))
    print('Generated: {}'.format(sample_sent))

    print()

    # Interpolation
    c = model.sample_c_prior(1)

    z1 = model.sample_z_prior(1).view(1, 1, z_dim)
    z1 = z1.cuda() if args.gpu else z1

    z2 = model.sample_z_prior(1).view(1, 1, z_dim)
    z2 = z2.cuda() if args.gpu else z2

    # Interpolation coefficients
    alphas = np.linspace(0, 1, 5)

    print('Interpolation of z:')
    print('-------------------')

    for alpha in alphas:
        z = float(1-alpha)*z1 + float(alpha)*z2

        sample_idxs = model.sample_sentence(z, c, temp=0.3)
        sample_sent = ' '.join([TEXT.vocab.itos[i] for i in sample_idxs])

        print("{}".format(sample_sent))

    print()
