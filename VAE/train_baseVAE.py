import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
from torchtext import data


class RNN_VAE(nn.Module):
    """
    1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015).
    3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, n_vocab, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.p_word_dropout = p_word_dropout

        self.gpu = gpu

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim+z_dim, z_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(z_dim, n_vocab)


        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.q_mu.parameters(),
            self.q_logvar.parameters()
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        )

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_params, self.decoder_params
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """    
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        return z

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x (z_dim+c_dim)
        init_h = z.unsqueeze(0)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y


    def forward(self, sentence):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z)       
        
        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def generate_sentences(self, batch_size):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            samples.append(self.sample_sentence(z, raw=True))

        return samples

    def sample_sentence(self, z, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z= z.view(1, 1, -1)

        h = z

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y/temp, dim=0)

            idx = torch.multinomial(y,1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, mbsize, temp=1):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        samples = []
        targets_z = []

        for _ in range(mbsize):
            z = self.sample_z_prior(1)

            samples.append(self.sample_soft_embed(z, temp=1))
            targets_z.append(z)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)

        return X_gen, targets_z

    def sample_soft_embed(self, z, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z = z.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z], 2)

        h = z

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.word_emb(word).view(1, -1)]

        for i in range(self.MAX_SENT_LEN):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
    
'''
###########################################################################
mTEXT = data.Field(tokenize='spacy')
mLABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset male_sent_obftrain_less700.tsv...")
mtrain  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='male_sent_obftrain_less700.tsv',
        format='tsv',
        fields=[('Text', mTEXT),('Label', mLABEL)])[0]

print("creating vocab for mTEXT")
mTEXT.build_vocab(mtrain, max_size=60000, vectors="glove.6B.100d")
mLABEL.build_vocab(mtrain)

mLABEL.vocab.stoi['1']=1
mLABEL.vocab.stoi['2']=2
mLABEL.vocab.stoi['3']=3
mLABEL.vocab.stoi['4']=4
mLABEL.vocab.stoi['5']=5


#################################
fTEXT = data.Field(tokenize='spacy')
fLABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset female_sent_obftrain_less700.tsv...")
ftrain  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='female_sent_obftrain_less700.tsv',
        format='tsv',
        fields=[('Text', fTEXT),('Label', fLABEL)])[0]

print("creating vocab for fTEXT")
fTEXT.build_vocab(ftrain, max_size=60000, vectors="glove.6B.100d")
fLABEL.build_vocab(ftrain)

fLABEL.vocab.stoi['1']=1
fLABEL.vocab.stoi['2']=2
fLABEL.vocab.stoi['3']=3
fLABEL.vocab.stoi['4']=4
fLABEL.vocab.stoi['5']=5

###############################
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):       
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))        
        y = self.fc(hidden.squeeze(0))
        return y


criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
criterion = criterion.to(device)


#frnn = torch.load('frnn8', map_location=lambda storage, loc: storage) #force to load on CPU
#mrnn = torch.load('mrnn8', map_location=lambda storage, loc: storage) #force to load on CPU
print("loading frnn model and mrnn model....")
frnn = torch.load('../sent_analysis/frnn8')
mrnn = torch.load('../sent_analysis/mrnn8')
mrnn = mrnn.to(device)
frnn = frnn.to(device)
criterion = criterion.to(device)


import spacy
nlp = spacy.load('en')
def predict_sentiment(sentence,model,TEXT):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    model.eval()
    prediction = model(tensor)
    return prediction.item()


def predict_word(word):
    print('frnn: ',predict_sentiment(word,frnn,fTEXT))
    print('mrnn:  ',predict_sentiment(word,mrnn,mTEXT))
    
    
###########################################################################   
    
'''    
    
    
    
    
    
    
    
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import *
import argparse



mb_size = 64
h_dim = 128
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = 128

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
from torchtext import data

#TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy',fix_length=20)
TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
LABEL = data.Field(sequential=False, unk_token=None)

print("loading dataset all_sent_trainVAE_less100.tsv...")
train  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='all_sent_trainVAE_less100.tsv',
        format='tsv',
        fields=[('Text', TEXT),('Label', LABEL)])[0]

TEXT.build_vocab(train, max_size=100000, vectors="fasttext.en.300d")
#TEXT.build_vocab(train, max_size=10000)
LABEL.build_vocab(train)


LABEL.vocab.stoi['1']=1
LABEL.vocab.stoi['2']=2
LABEL.vocab.stoi['3']=3
LABEL.vocab.stoi['4']=4
LABEL.vocab.stoi['5']=5


model = RNN_VAE(
    len(TEXT.vocab), h_dim, z_dim, p_word_dropout=0.3,max_sent_len=40,
    pretrained_embeddings=TEXT.vocab.vectors, freeze_embeddings=False,
    gpu=True
)

# Annealing for KL term
kld_start_inc = 3000
kld_weight = 0.01
kld_max = 0.15
kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

trainer = optim.Adam(model.vae_params, lr=lr)

train_iter = data.BucketIterator(
dataset=train, batch_size=mb_size,
sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))


#print("loading previous 200yelp_nofix_max40_predif.bin model")
#model.load_state_dict(torch.load('models/{}.bin'.format('pre_yelp128_100')))
    
def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format('pre_yelp128_all_sent_trainVAE_less100'))   
    
# pre_yelp128_all_sent_trainVAE_less100.bin 是用 all_sent_trainVAE_less100.tsv 训练的base_VAE model, 词汇max=100000

    
#pre_weight = 1000
    
#record=[]    

for it in range(400000):
    batch = next(iter(train_iter))
    inputs = batch.Text
    labels = batch.Label

    recon_loss, kl_loss = model.forward(inputs)
    loss = (recon_loss + kld_weight * kl_loss)#*(pre_weight*pre_dif)
    #print("pre_weight*pre_dif: ",pre_weight*pre_dif)

    # Anneal kl_weight
    if it > kld_start_inc and kld_weight < kld_max:
        kld_weight += kld_inc

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
    trainer.step()
    trainer.zero_grad()

    #if it % log_interval == 0:
    if it%100==0:
        #original_sent = ' '.join([TEXT.vocab.itos[i] for i in inputs[:,0][1:]])
        #m = predict_sentiment(original_sent,mrnn,mTEXT)
        #f = predict_sentiment(original_sent,frnn,fTEXT)
        #print(original_sent)
        #print("mrnn original prediction: ",m)
        #print("frnn original prediction: ",f)
        #print("abs original dif: ",abs(m-f))
        
        
        z = model.sample_z_prior(1)
        sample_idxs = model.sample_sentence(z)
        sample_sent = ' '.join([TEXT.vocab.itos[i] for i in sample_idxs])
        
        print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
              .format(it, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm))
        print('Sample: "{}"'.format(sample_sent))
        
        if sample_sent:
            m0 = predict_sentiment(sample_sent,mrnn,mTEXT)
            f0 = predict_sentiment(sample_sent,frnn,fTEXT)
            pre_dif_sample = abs(f0-m0)
            record.append([pre_dif_sample,sample_sent])
            print("mrnn sample prediction: ",m0)
            print("frnn sample prediction: ",f0)
            print("sample abs dif: ",abs(m0-f0))
            print("\n")

        

    # Anneal learning rate
    new_lr = lr * (0.5 ** (it // lr_decay_every))
    for param_group in trainer.param_groups:
        param_group['lr'] = new_lr
        
    if it%1000==0:
        print("saving model pre_yelp128_all_sent_trainVAE_less100.bin")
        print("\n")
        save_model()

'''        
with open('baseVAE_predif_sample','w') as f:
    for x in record:
        f.write(str(x))
        f.write('\n')
'''        
        
    
