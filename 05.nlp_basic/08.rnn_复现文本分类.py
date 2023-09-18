import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn


def read_data(file, num=None):
    with open(file, encoding='utf-8') as f:
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
    if num is not None:
        return all_text[:num], all_label[:num]
    return all_text, all_label


def get_word_2_index(all_text):
    word_2_index = {"PAD":0,"UNK":1}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


class MyDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        label = self.all_label[index]
        text_idx = [word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (max_len - len(text_idx))

        return torch.tensor(text_idx), torch.tensor(label)

    def __len__(self):
        assert len(self.all_text) == len(self.all_label)
        return len(self.all_text)


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, emb_size = x.shape
        t = torch.zeros((1, self.hidden_size), device=x.device)
        result = torch.zeros(size=(batch_size, seq_len, self.hidden_size), device=x.device)

        for i in range(seq_len):
            w_emb = x[:, i]
            h1 = self.W.forward(w_emb)
            h2 = h1 + t
            h3 = self.tanh(h2)

            t = self.U.forward(h3)

            result[:, i] = h3
        return result, t


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding(word_size, emb_dim)
        self.rnn = MyRNN(emb_dim, rnn_hidden_num, batch_first=True)
        self.cls = nn.Linear(rnn_hidden_num, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None):  # 10 * 30 * 200 @ 200 * 400 = 10 * 30 * 400
        x = self.emb.forward(x)  # 10 * 30 ---> 10 * 30 * emb_num
        rnn_out1, rnn_out2 = self.rnn.forward(x)  # 10 * 30 * emb_num ----> 10 * 30 * hidden_num
        rnn_out1 = rnn_out1[:, -1]  # 0, -1, mean, max
        pre = self.cls(rnn_out1)

        if label is not None:
            loss = self.loss_fun(pre, label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..', 'data', '文本分类', 'train.txt'))
    test_text, test_label = read_data(os.path.join('..', 'data', '文本分类', 'test.txt'))
    word_2_index, index_2_word = get_word_2_index(train_text)

    epoch = 10
    max_len = 30
    batch_size = 10
    word_size = len(word_2_index)
    emb_dim = 200
    class_num = 10
    rnn_hidden_num = 100
    lr = 0.00003
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    test_dataset = MyDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = MyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_text_idx, batch_label in train_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_label = batch_label.to(device)

            loss = model.forward(batch_text_idx, batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

        print(f"loss: {loss}")

        right_num = 0
        for batch_text_idx, batch_label in test_dataloader:
            batch_text_idx = batch_text_idx.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_text_idx)
            right_num += int(torch.sum(pre == batch_label))

        acc = right_num / len(test_dataset)
        print(f"acc: {acc:.3f}")