import os

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


def read_data(path, num=None):
    with open(path, encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []

    for data in all_data:
        s_data = data.split('\t')
        if len(s_data) != 2:
            continue
        text, label = s_data
        all_text.append(text)
        all_label.append(int(label))

    if num is None:
        return all_text, all_label
    else:
        return all_text[:num], all_label[:num]


def get_word_2_index(all_text):
    word_2_index = {'PAD': 0, 'UNK': 1}
    for text in all_text:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]

        text_idx = [word_2_index.get(w, 1) for w in text]
        text_idx += [0] * (max_len - len(text_idx))

        return torch.tensor(text_idx), torch.tensor(label)

    def __len__(self):
        return len(self.all_label)


class MyLSTM(nn.Module):
    def __init__(self, emb_num, lstm_hidden_num, batch_first=True):
        super().__init__()
        self.lstm_hidden_num = lstm_hidden_num
        self.emb_num = emb_num

        self.F = nn.Linear(emb_num + lstm_hidden_num, lstm_hidden_num)
        self.I = nn.Linear(emb_num + lstm_hidden_num, lstm_hidden_num)
        self.C = nn.Linear(emb_num + lstm_hidden_num, lstm_hidden_num)
        self.O = nn.Linear(emb_num + lstm_hidden_num, lstm_hidden_num)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, emb_n = x.shape

        a_prev = torch.zeros((batch_size, self.lstm_hidden_num), device=x.device, requires_grad=False)
        c_prev = torch.zeros((batch_size, self.lstm_hidden_num), device=x.device, requires_grad=False)

        result = torch.zeros(batch_size, seq_len, self.lstm_hidden_num, device=x.device)
        for wi in range(seq_len):
            w_emb = x[:, wi]
            w_a_emb = torch.cat((w_emb, a_prev), dim=-1)

            f = self.F(w_a_emb)
            i = self.I(w_a_emb)
            c = self.C(w_a_emb)
            o = self.O(w_a_emb)

            ft = self.sigmoid(f)
            it = self.sigmoid(i)
            cct = self.tanh(c)
            ot = self.sigmoid(o)

            c_next = ft * c_prev + it * cct

            th = self.tanh(c_next)
            a_next = th * ot

            a_prev = a_next
            c_prev = c_next

            result[:, wi] = a_next

        # return result,(a_next,c_next)
        return result, (a_prev, c_prev)


class RNNTextCls(nn.Module):
    def __init__(self, word_size, emb_dim, hidden_num, class_num, bi):
        super().__init__()
        self.emb = nn.Embedding(word_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_num, batch_first=True, bidirectional=bi, num_layers=1)
        if bi:
            self.cls = nn.Linear(hidden_num * 2, class_num)
        else:
            self.cls = nn.Linear(hidden_num, class_num)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        emb = self.emb.forward(x)  # x : 10 * 30 --> 10 * 30 * emb_num
        lstm_out1, lstm_out2 = self.rnn.forward(emb)
        # 10 * 30 * emb_num ---> 10 * 30 * hidden

        rnn_out1 = lstm_out1[:, -1]  # 0 , -1 , mean , max
        pre = self.cls(rnn_out1)

        if label is not None:
            loss = self.loss_fn(pre, label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'), 50000)
    test_text, test_label = read_data(os.path.join('..', 'data', '文本分类', 'test.txt'), 400)

    word_2_index, index_2_word = get_word_2_index(train_text)

    batch_size = 20
    epoch = 10
    max_len = 30
    word_size = len(word_2_index)
    emb_dim = 200
    hidden_num = 300
    class_num = 10
    lr = 0.0002
    bi = True

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    test_dataset = MyDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNTextCls(word_size, emb_dim, hidden_num, class_num, bi).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)

    for e in range(epoch):
        model.train()
        for batch_text_idx, batch_lable in train_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_lable = batch_lable.to(device)

            loss = model.forward(batch_text_idx, batch_lable)
            loss.backward()

            opt.step()
            opt.zero_grad()

        model.eval()
        right_num = 0
        for batch_text_idx, batch_lable in test_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_lable = batch_lable.to(device)

            pre = model.forward(batch_text_idx)

            right_num += int(torch.sum(pre == batch_lable))

        acc = right_num / len(test_dataset)
        print(acc)