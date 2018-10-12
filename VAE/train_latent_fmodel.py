import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
from torchtext import data


# male_sent_obftrain_less100.tsv has total 9610 reviews 712KB
# female_sent_obfandclass_less100.tsv has total 9707 reviews 723KB
# 上面两个数据集加在一起：all_sent_trainVAE_less100.tsv


class Net(nn.Module):

    def __init__(self,z_dim,dropout):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(z_dim, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)  
        return x


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
        z = mu + torch.exp(logvar/2) * eps
        
        
        

        
        return z/z.pow(2).sum().pow(0.5)

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        
        return z/z.pow(2).sum().pow(0.5)

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
    
##############################################
# creating female vocabuary from female_sent_obfandclass_less100.tsv

fTEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
fLABEL = data.Field(sequential=False, unk_token=None)

#fTEXT = data.Field(tokenize='spacy')
#fLABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset female_sent_obfandclass_less100.tsv...")
ftrain  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='female_sent_obfandclass_less100.tsv',
        format='tsv',
        fields=[('Text', fTEXT),('Label', fLABEL)])[0]

fTEXT.build_vocab(ftrain, max_size=100000, vectors="fasttext.en.300d")
fLABEL.build_vocab(ftrain)

fLABEL.vocab.stoi['1']=float(1)
fLABEL.vocab.stoi['2']=float(2)
fLABEL.vocab.stoi['3']=float(3)
fLABEL.vocab.stoi['4']=float(4)
fLABEL.vocab.stoi['5']=float(5)

################################################
# creating male&female vocabulary from all_sent_trainVAE_less100.tsv 
#(这个包含了male_sent_obftrain_less100.tsv和 female_sent_obfandclass_less100.tsv)

allTEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
allLABEL = data.Field(sequential=False, unk_token=None)

#allTEXT = data.Field(tokenize='spacy')
#allLABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset all_sent_trainVAE_less100.tsv...")
alltrain  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='all_sent_trainVAE_less100.tsv',
        format='tsv',
        fields=[('Text', allTEXT),('Label', allLABEL)])[0]

allTEXT.build_vocab(alltrain, max_size=100000, vectors="fasttext.en.300d")
allLABEL.build_vocab(alltrain)

allLABEL.vocab.stoi['1']=float(1)
allLABEL.vocab.stoi['2']=float(2)
allLABEL.vocab.stoi['3']=float(3)
allLABEL.vocab.stoi['4']=float(4)
allLABEL.vocab.stoi['5']=float(5)

#################################################
fTEXT.vocab = allTEXT.vocab

mb_size = 32
h_dim = 128
lr = 1e-3
lr_decay_every = 2000000
n_iter = 20000
log_interval = 1000
z_dim = 128

# VAE model trained on all_sent_trainVAE_less100.tsv
VAEmodel = RNN_VAE(
    len(allTEXT.vocab), h_dim, z_dim, p_word_dropout=0.3,max_sent_len=40,
    pretrained_embeddings=allTEXT.vocab.vectors, freeze_embeddings=False,
    gpu=True
)

VAEmodel.load_state_dict(torch.load('models/{}.bin'.format('pre_yelp128_all_sent_trainVAE_sph_less100'))) #VAE model

def get_codez(inputs):
    #VAEmodel.eval()

    # sentence: '<start> I want to fly <eos>'
    # enc_inputs: '<start> I want to fly <eos>'
    # dec_inputs: '<start> I want to fly <eos>'
    # dec_targets: 'I want to fly <eos> <pad>'
    pad_words = Variable(torch.LongTensor([VAEmodel.PAD_IDX])).repeat(1, mb_size)
    pad_words = pad_words.cuda() if VAEmodel.gpu else pad_words

    enc_inputs = inputs
    dec_inputs = inputs
    dec_targets = torch.cat([inputs[1:], pad_words], dim=0)

    # Encoder: sentence -> inputs -> z
    mu, logvar = VAEmodel.forward_encoder(enc_inputs)
    z = VAEmodel.sample_z(mu, logvar)
    return z

'''
# 不带validation的 (可以无限for循环下去)
ftrain_iter = data.BucketIterator(
dataset=ftrain, batch_size=mb_size,
sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
'''

# ## 带有validation的 (循环完一遍就会停止,所以要自己设置循环停止的iteration)
ftrain_split, fvalid_split = ftrain.split(split_ratio=0.8)
ftrain_split_iter = data.BucketIterator(dataset=ftrain_split, batch_size=mb_size,sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
fvalid_split_iter = data.BucketIterator(dataset=fvalid_split, batch_size=mb_size,sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))






fmodel = Net(z_dim,0.5)
print("fmodel parameters: ")
print(fmodel.parameters)

#pretrained_embeddings = fTEXT.vocab.vectors
#fmodel.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
foptimizer = optim.Adam(fmodel.parameters(),lr=0.0003)
fcriterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
fmodel = fmodel.to(device)
fcriterion = fcriterion.to(device)

import torch.nn.functional as F

def accuracy(preds,y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds==y).float()
    acc = correct.sum()/len(correct)
    return acc



def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train() # turns on dropout and batch normalization and allow gradient update
    
    i=0
    for batch in iterator:
        i=i+1
        if i>10000:
            break
        
        inputs = batch.Text
        labels = batch.Label
    
        try:
            z = get_codez(inputs)
        except:
            print('pass training')
            continue
        
        optimizer.zero_grad() # set accumulated gradient to 0 for every start of a batch
        
        predictions = model(z).squeeze(1)
        
        labels = torch.cuda.FloatTensor([x for x in labels])
        
        labels = labels.to(device)
        
        loss = criterion(predictions, labels)
        
        acc = accuracy(predictions, labels)
        
        loss.backward() # calculate gradient
        
        optimizer.step() # update parameters
        
        if i%200==0:
            print("train batch loss: ", loss.item())
            print("train accuracy: ", acc.item())
            print(i)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss / 10000, epoch_acc / 10000


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval() #turns off dropout and batch normalization
    
    with torch.no_grad():
        i=0
        for batch in iterator:
            i=i+1
            if i>len(iterator):
                break
            
            inputs = batch.Text
            labels = batch.Label

            try:
                z = get_codez(inputs)
            except:
                print('pass evaluation')
                continue
        
            predictions = model(z).squeeze(1)
        
            labels = torch.cuda.FloatTensor([x for x in labels])

            labels = labels.to(device)

            loss = criterion(predictions, labels)

            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if i%50 ==0:
                print("eval batch loss: ", loss.item())
                print("eval accuracy: ", acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#model = torch.load('fmodel')

import timeit
#start = timeit.default_timer()


###########################################
# start training

N_EPOCHS = 20
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(fmodel, ftrain_split_iter, foptimizer, fcriterion)
        valid_loss, valid_acc = evaluate(fmodel, fvalid_split_iter, fcriterion)
        #print("saving model:   frnn8")
        #torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)


except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')
    