import numpy as np

from IMDB import IMDB
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import tqdm

from matplotlib import pyplot as plt
import pandas as pd

torch.manual_seed(1234)
print("cuda: ", torch.cuda.is_available())
mm = IMDB()
train, trainlabel, test, testlabel, lexicon = mm.get_datasets()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")


# print(lexicon)
print("Vocab length: " + str(len(lexicon)))
#
# row_len = [len(row) for row in train]
# pd.Series(row_len).hist()
# plt.savefig("fig 3")
# print(pd.Series(row_len).describe())
#

# padding the sequences such that there is a maximum length of num
def transformToFeatures(data, num):
    features = np.zeros((len(data), num), dtype=int)
    for i, row in enumerate(data):
        if len(row) != 0:
            features[i, -len(row):] = np.array(row)[:num]
    return features

features_train = transformToFeatures(train, 800)
features_test = transformToFeatures(test, 800)
print(features_train)
print(np.shape(features_train))

train_tensor = TensorDataset(torch.from_numpy(features_train), torch.from_numpy(trainlabel))
test_tensor = TensorDataset(torch.from_numpy(features_test), torch.from_numpy(testlabel))

batch_size = 16
train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_tensor, shuffle=True, batch_size=batch_size, drop_last=True)

iterator = iter(train_loader)
sample_x, sample_y = iterator.next()

print(sample_x.size())
print(sample_x)
print(sample_y)

class LstmRnn(nn.Module):
    def __init__(self, num_layers, lexicon_size, hid_size, out_size, emb_size, drop_prob=0.1):
        super(LstmRnn, self).__init__()

        self.out_size = out_size
        self.hid_size = hid_size

        self.num_layers = num_layers
        self.lexicon_size = lexicon_size

        # embedding layer
        self.emb = nn.Embedding(lexicon_size, emb_size)

        # lstm layer
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=self.hid_size, num_layers=round(num_layers/2), batch_first=True, bidirectional=True)

        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(drop_prob)

        # linear layer
        self.linear = nn.Linear(self.hid_size*2, out_size)

        # sigmoid layer(output)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hid_size).to(device)
        # initialize cell state
        c0 = torch.zeros(self.num_layers, batch_size, self.hid_size).to(device)
        hid = (h0, c0)
        return hid
    # forward pass of the algorithm
    def forward(self, features, hid):
        batch_size = features.size(0)

        # pass through embedding layer
        embeds = self.emb(features) # shape: B x S x Feature

        # pass through lstm layer
        lstm_ret, hid = self.lstm(embeds, hid)

        # pass through dropout layer
        ret = self.dropout(lstm_ret)

        # pass through linear layer
        ret = self.linear(ret)

        # pass through sigmoid
        sig_ret = self.sigmoid(ret)

        # reshape
        sig_ret = sig_ret.view(batch_size, -1)
        # get last labels batch
        sig_ret = sig_ret[:, -1]

        return sig_ret, hid

num_layers = 4
lexicon_size = len(lexicon) + 1     # pad by 1
emb_size = 1000
out_size = 1
hid_size = 512
clip = 5
epochs = 30

model = LstmRnn(num_layers, lexicon_size, hid_size, out_size, emb_size, drop_prob=0.5)

# if gpu can be used, then use gpu otherwise cpu
model.to(device)

print(model)

learning_rate = 0.0001
criterion = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

def acc(prediction, actual):
    prediction = torch.round(prediction.squeeze())
    return torch.sum(prediction == actual.squeeze()).item()

val_loss_min = np.Inf

etrain_losses, eval_losses = [], []
etrain_acc, eval_acc = [], []

for epoch in tqdm.tqdm(range(epochs)):
    # learning_rate*=0.9
    train_losses = []
    tracc = 0.0
    model.train()
    h = model.init_hidden(batch_size)
    print("Training...")
    for features, labels in tqdm.tqdm(train_loader):
        # create new variables to prevent backpropagating through the whole history
        h = tuple([x.data for x in h])
        model.zero_grad()
        features, labels = features.to(device), labels.to(device)

        out, h = model(features, h)

        # find loss

        loss = criterion(out.squeeze(), labels.float())

        # backprop itself
        loss.backward()
        train_losses.append(loss.item())

        accuracy = acc(out, labels)
        tracc += accuracy

        # prevent exploding grad
        nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()

    val_h = model.init_hidden(batch_size)
    v_losses = []
    val_acc = 0.0
    model.eval()

    print("validating...")
    for features, labels in tqdm.tqdm(test_loader):
        val_h = tuple([x.data for x in val_h])

        features, labels = features.to(device), labels.to(device)
        out, val_h = model(features, val_h)

        v_loss = criterion(out.squeeze(), labels.float())
        v_losses.append(v_loss.item())

        accuracy = acc(out, labels)
        val_acc += accuracy

    etrain_loss = np.mean(train_losses)
    eval_loss = np.mean(v_losses)

    etr_acc = tracc/len(train_loader.dataset)
    ev_acc = val_acc/len(test_loader.dataset)

    etrain_losses.append(etrain_loss)
    eval_losses.append(eval_loss)

    etrain_acc.append(etr_acc)
    eval_acc.append(ev_acc)
    print(f'train_loss : {etrain_loss} val_loss : {eval_loss}')
    print(f'train_acc : {etr_acc} val_acc : {ev_acc}')
    if(eval_loss <= val_loss_min):
        torch.save(model.state_dict(), 'state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}. Saving model ...)'.format(val_loss_min, eval_loss))
        val_loss_min = eval_loss
    print(25*'==')

    fig = plt.figure(figsize=(20,6))
    plt.subplot(1, 2, 1)
    plt.plot(etrain_acc, label='Training accuracy')
    plt.plot(eval_acc, label='Validation accuracy')
    plt.title("Accuracies")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(etrain_losses, label='Training loss')
    plt.plot(eval_losses, label='Validation loss')
    plt.title("Losses")
    plt.legend()
    plt.grid()
    plt.savefig("performance")

    plt.show()
