import torch
from torchtext import data
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset female_sent_obftrain_less700.tsv...")
train  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/', 
        train='female_sent_obftrain_less700.tsv',
        format='tsv',
        fields=[('Text', TEXT),('Label', LABEL)])[0]

TEXT.build_vocab(train, max_size=60000, vectors="fasttext.en.300d")
LABEL.build_vocab(train)

LABEL.vocab.stoi['1']=1
LABEL.vocab.stoi['2']=2
LABEL.vocab.stoi['3']=3
LABEL.vocab.stoi['4']=4
LABEL.vocab.stoi['5']=5


import json
with open('female_sent_tmp_train_vocab_frnn8','w') as f:
    json.dump(TEXT.vocab.stoi,f)




BATCH_SIZE = 32

train, valid = train.split(split_ratio=0.999)
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)
'''
train_iterator = data.BucketIterator.splits(
    train, 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text), 
    repeat=False)
'''


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(x))
        #print("embedded shape: ", embedded.shape)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        #print("output.shape: ",output.shape)
        #print("output[-1].shape: ",output[-1].shape)
        #print("hidden.shape: ",hidden.shape)
        #print("cell.shape: ",cell.shape)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid. dim]
        #cell = [num layers * num directions, batch size, hid. dim]
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #print("hidden.shape: ",hidden.shape)
        
        y = self.fc(hidden.squeeze(0))
                
        #hidden [batch size, hid. dim * num directions]
            
        #return self.fc(hidden.squeeze(0))
        return y


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 500
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
print("model parameters: ")
print(model.parameters)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(),lr=0.0003)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
model = model.to(device)
criterion = criterion.to(device)

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
        
        optimizer.zero_grad() # set accumulated gradient to 0 for every start of a batch
        
        predictions = model(batch.Text).squeeze(1)
        
        loss = criterion(predictions, batch.Label)
        
        acc = accuracy(predictions, batch.Label)
        
        loss.backward() # calculate gradient
        
        optimizer.step() # update parameters
        
        if i%400==0:
            print("train batch loss: ", loss.item())
            print("train accuracy: ", acc.item())
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval() #turns off dropout and batch normalization
    
    with torch.no_grad():
        i=0
        for batch in iterator:
            i=i+1
            predictions = model(batch.Text).squeeze(1)
            
            loss = criterion(predictions, batch.Label)
            
            acc = accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if i%200 ==0:
                print("eval batch loss: ", loss.item())
                print("eval accuracy: ", acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#model = torch.load('fmodel')

import timeit
#start = timeit.default_timer()


N_EPOCHS = 20
#print("loading previous frnn3 model...")
#model = torch.load('frnn3')
try:
    for epoch in range(N_EPOCHS):
        start = timeit.default_timer()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print("saving model:   frnn8")
        torch.save(model,'frnn8')

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

        stop = timeit.default_timer()
        print("time duration:    ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print("save frnn8 again:")
torch.save(model,'frnn8')


####################
# prediction
####################

'''
print('loading frnn4:')
model = torch.load('frnn4',map_location=lambda storage,loc:storage)
'''



valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
print("valid loss: ",valid_loss)
print("valid acc: ",valid_acc)

    
print("prediction of frnn8.....")
    
import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence,model):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    model.eval()
    prediction = model(tensor)
    return prediction.item()


with open('../sent/ori_gender_data/male_sent_test_less700.tsv','r') as f:
    mtest = f.readlines()

with open('../sent/ori_gender_data/female_sent_test_less700.tsv','r') as f:
    ftest = f.readlines()

fs = [line.split('\t')[0] for line in ftest]
ms = [line.split('\t')[0] for line in mtest]

mlabel = [int(line.split('\t')[1].strip('\n')) for line in mtest]
flabel = [int(line.split('\t')[1].strip('\n')) for line in ftest]

fprem = [predict_sentiment(x,model) for x in ms]
fpref = [predict_sentiment(x,model) for x in fs]

print("10 fprem:")
print(fprem[:10])
print("10 fpref:")
print(fpref[:10])
     
      
print("writing fpref to file fpref_frnn8.txt...")
with open('fpref_frnn8.txt','w') as f:
    f.write(str(fpref))
print("writing fprem to file fprem_frnn8.txt...")
with open('fprem_frnn8.txt','w') as f:
    f.write(str(fprem))

print("fpref accuracy:    ",(np.array([round(x) for x in fpref])==np.array(flabel)).mean())
print("fprem accuracy:    ",(np.array([round(x) for x in fprem])==np.array(mlabel)).mean())


'''
with open('../sent/ori_gender_data/male_sent_tmp_train.tsv','r') as f:
    mtrain = f.readlines()

with open('../sent/ori_gender_data/female_sent_tmp_train.tsv','r') as f:
    ftrain = f.readlines()

fs = [line.split('\t')[0] for line in ftrain]
ms = [line.split('\t')[0] for line in mtrain]

mlabel = [int(line.split('\t')[1].strip('\n')) for line in mtrain]
flabel = [int(line.split('\t')[1].strip('\n')) for line in ftrain]

fprem = [predict_sentiment(x,model) for x in ms]
fpref = [predict_sentiment(x,model) for x in fs]

print("10 fpref on female_sent_tmp_train.tsv:")
print(fpref[:10])
print("10 fprem on male_sent_tmp_train.tsv:")
print(fprem[:10])
     
      
print("writing fpref to file :fpre_female_sent_tmp_train_frnn4.txt...")
with open('fpre_female_sent_tmp_train_frnn4.txt','w') as f:
    f.write(str(fpref))
print("writing fprem to file :fpre_male_sent_tmp_train_frnn4.txt...")
with open('fpre_male_sent_tmp_train_frnn4.txt','w') as f:
    f.write(str(fprem))


print("fpref accuracy:    ",(np.array([round(x) for x in fpref])==np.array(flabel)).mean())
print("fprem accuracy:    ",(np.array([round(x) for x in fprem])==np.array(mlabel)).mean())
'''



