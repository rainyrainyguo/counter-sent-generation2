import torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

print("loading dataset male_sent_tmp_train.tsv")
train  = data.TabularDataset.splits(
        path='../sent/ori_gender_data/',
        train='male_sent_tmp_train.tsv',
        format='tsv',
        fields=[('Text', TEXT),('Label', LABEL)])[0]

TEXT.build_vocab(train, max_size=50000, vectors="glove.6B.100d")
LABEL.build_vocab(train)

train, valid = train.split()


LABEL.vocab.stoi['1']=1
LABEL.vocab.stoi['2']=2
LABEL.vocab.stoi['3']=3
LABEL.vocab.stoi['4']=4
LABEL.vocab.stoi['5']=5

f = open('male_sent_tmp_train_vocab_mcnn3.txt','w')
for item in TEXT.vocab.stoi.items():
    f.write(item[0]+','+ str(item[1]) +'\n')
f.close()


BATCH_SIZE = 64

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train, valid),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.Text),
    repeat=False)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]

        x = x.permute(1, 0)

        # x = [batch size, sent len]

        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
print("model parameters: ")
print(model.parameters)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

def accuracy(preds,y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds==y).float()
    acc = correct.sum()/len(correct)
    return acc


def trains(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()  # turns on dropout and batch normalization and allow gradient update

    i = 0
    for batch in iterator:
        i = i + 1

        optimizer.zero_grad()  # set accumulated gradient to 0 for every start of a batch

        predictions = model(batch.Text).squeeze(1)

        loss = criterion(predictions, batch.Label)

        acc = accuracy(predictions, batch.Label)

        loss.backward()  # calculate gradient

        optimizer.step()  # update parameters

        if i % 200 == 0:
            print("train batch loss: ", loss.item())
            print("train accuracy: ", acc.item())

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()  # turns off dropout and batch normalization

    with torch.no_grad():
        i = 0
        for batch in iterator:
            if batch.Text.shape[0]<5:
                continue
            i = i + 1
            predictions = model(batch.Text).squeeze(1)

            loss = criterion(predictions, batch.Label)

            acc = accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if i % 200 == 0:
                print("eval batch loss: ", loss.item())
                print("eval accuracy: ", acc.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import torch.nn.functional as F
import timeit

start = timeit.default_timer()

N_EPOCHS = 30
try:
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = trains(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print("saving model...mcnn3")
        torch.save(model,'mcnn3')
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')
        #print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%')

    stop = timeit.default_timer()
    print("time duration:  ", stop - start)

except KeyboardInterrupt:
    print("interrupt")
    print('Exiting from training early')

print("save mcnn3 again:")
torch.save(model,'mcnn3')

####################
# prediction
####################

import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = model(tensor)
    return prediction.item()

with open('../sent/ori_gender_data/male_sent_test.tsv','r') as f:
    mtest = f.readlines()
with open('../sent/ori_gender_data/female_sent_test.tsv','r') as f:
    ftest = f.readlines()

mlabel = [int(line.split('\t')[1].strip('\n')) for line in mtest]
flabel = [int(line.split('\t')[1].strip('\n')) for line in ftest]

ms=[]
fs=[]
for i in range(len(mtest)):
    if len(mtest[i].split('\t')[0].split())<5:
        ms.append(mtest[i].split('\t')[0]+' <pad>'+' <pad>'+' <pad>'+' <pad>')
    else:
        ms.append(mtest[i].split('\t')[0])

for i in range(len(ftest)):
    if len(ftest[i].split('\t')[0].split())<5:
        fs.append(ftest[i].split('\t')[0]+' <pad>'+' <pad>'+' <pad>'+' <pad>')
    else:
        fs.append(ftest[i].split('\t')[0])

mpref = [predict_sentiment(x) for x in fs]
mprem = [predict_sentiment(x) for x in ms]

print("writing mpref to file...")
with open('mpref_mcnn3.txt', 'w') as f:
    f.write(mpref)
print("writing mprem to file...")
with open('mprem_mcnn3.txt', 'w') as f:
    f.write(mprem)

print("mpref accuracy:    ", (np.array([round(x) for x in mpref]) == np.array(flabel)).mean())
print("mprem accuracy:    ", (np.array([round(x) for x in mprem]) == np.array(mlabel)).mean())

