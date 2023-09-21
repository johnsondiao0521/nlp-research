import torch
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset, DataLoader


def read_data(file, num=None):
    with open(file, encoding='utf8') as f:
        all_data = f.read().split('\n')
    if num:
        return all_data[:num]
    return all_data


def get_word_2_index(all_text):
    word_2_index = {"PAD": 0, "UNK": 1, "STA": 2, "END": 3}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))
    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


class PDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __getitem__(self, index):
        text = self.all_data[index]
        input_text = text[:-1]
        label_text = text[1:]

        input_idx = [2] + [word_2_index.get(i, 1) for i in input_text]
        label_idx = [word_2_index.get(i, 1) for i in label_text] + [3]
        return torch.tensor(input_idx), torch.tensor(label_idx)

    def __len__(self):
        return len(self.all_data)


class PModel(nn.Module):
    def __init__(self):
        super(PModel, self).__init__()
        self.emb = nn.Embedding(word_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, rnn_hidden_num, batch_first=True)
        self.cls = nn.Linear(rnn_hidden_num, word_size)
        self.dropout = nn.Dropout(0.2)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None, h=None):
        x = self.emb(x)
        batch_size, seq_len, emb_dim = x.shape
        rnn_out1, rnn_out2 = self.rnn.forward(x, h)
        rnn_out1 = self.dropout(rnn_out1)
        pre = self.cls.forward(rnn_out1)

        if label is not None:
            loss = self.loss_fun(pre.reshape(batch_size * seq_len, -1), label.reshape(-1))
            return loss

        return torch.argmax(pre, dim=-1), rnn_out1


def auto_generate():
    result = ''
    word = "STA"
    h = None
    word_index = word_2_index[word]

    while True:
        word_index = torch.tensor([[word_index]])
        word_index, h = model.forward(word_index, h=h)
        # h = torch.squeeze(h, dim=0)
        word_index = int(word_index)

        if word_index == 3 or len(result) > 50:
            break

        result += index_2_word[word_index]
    return result


def acrostic_poem(text):
    text = text[:4]
    assert len(text) >= 4

    result = ""
    word = text[0]
    word_index = word_2_index[word]
    h = None

    for i in range(4):
        result += ''


if __name__ == '__main__':
    train_data = read_data(os.path.join('..', 'data', '古诗生成', 'poetry_5.txt'))
    word_2_index, index_2_word = get_word_2_index(train_data)

    epoch = 100
    batch_size = 10
    word_size = len(word_2_index)
    emb_dim = 100
    rnn_hidden_num = 50
    lr = 0.004

    train_dataset = PDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    model = PModel()
    opt = torch.optim.Adam(model.parameters(), lr)
    for e in range(epoch):
        for text_idx, label_idx in train_dataloader:
            loss = model.forward(text_idx, label_idx)
            loss.backward()

            opt.step()
            opt.zero_grad()
        print(f"loss: {loss:.3f}")

        result = auto_generate()
        print(f"result: {result}")